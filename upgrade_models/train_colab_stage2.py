import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from rb_lnl_ti import rb_lnl_ti
except ImportError:
    from upgrade_models.rb_lnl_ti import rb_lnl_ti

class IndexedGTSRB(datasets.GTSRB):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target, index

def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return train_transform, val_transform

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def prepare_sample_weights(csv_path, dataset_size):
    """
    Read the difficulty CSV and compute a weight tensor for Hard-example boosting.
    """
    print(f"Loading difficulty log from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Initialize all weights to 1.0
    weights_tensor = torch.ones(dataset_size)
    
    max_loss = df['loss'].max()
    df['normalized_loss'] = df['loss'] / (max_loss + 1e-8)
    
    alpha = 1.0
    w_max = 2.5
    
    # Formula: w_i = clip(1 + alpha * normalized_loss_i, 1, w_max)
    # Only boost Hard and Ambiguous samples as per the PDF
    mask = df['difficulty_group'].isin(['Hard', 'Ambiguous'])
    df.loc[mask, 'weight'] = np.clip(1 + alpha * df.loc[mask, 'normalized_loss'], 1.0, w_max)
    
    # Suspected noise can be clipped to 1.0 or lower (we keep 1.0)
    df.loc[~mask, 'weight'] = 1.0
    
    for _, row in df.iterrows():
        idx = int(row['image_id'])
        if idx < dataset_size:
            weights_tensor[idx] = row['weight']
            
    boosted_count = mask.sum()
    print(f"Boosted {boosted_count} samples (Hard/Ambiguous). Max weight applied: {weights_tensor.max().item():.2f}")
    return weights_tensor

def main():
    # Stage 2 config
    start_epoch = 90 # Technically epoch 91, but 0-indexed in loop range is 91 to 110
    end_epoch = 109
    epochs_stage2 = end_epoch - start_epoch + 1
    
    batch_size = 32
    learning_rate = 1e-4 # Lower LR for stage 2
    weight_decay = 0.04
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = './data'
    difficulty_csv_path = './submission/results/sample_difficulty_epoch_90.csv' # Output from Stage 1
    base_model_path = './submission/rb_lnl_ti_gtsrb.pth'
    checkpoint_path = './submission/latest_checkpoint_stage2.pth'
    best_model_path = './submission/rb_lnl_ti_gtsrb_stage2.pth'
    
    print(f"Using device: {device}")
    
    if not os.path.exists(difficulty_csv_path):
        print(f"ERROR: Cannot find {difficulty_csv_path}. Please run Stage 1 first!")
        return
        
    if not os.path.exists(base_model_path):
        print(f"ERROR: Cannot find base model {base_model_path}. Please run Stage 1 first!")
        return

    train_transform, val_transform = build_transforms()
    train_dataset, val_dataset = get_datasets(data_dir, train_transform, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Prepare weights
    sample_weights = prepare_sample_weights(difficulty_csv_path, len(train_dataset))
    
    # Model Setup
    model = rb_lnl_ti(num_classes=43, pretrained=False)
    model.use_residual = False # Stage 2 still uses base head only (Stage 3 activates residual)
    
    print(f"Loading base model weights from {base_model_path}...")
    model.load_state_dict(torch.load(base_model_path, map_location='cpu'))
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03, reduction='none') # None to apply weights manually
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage2)
    
    current_epoch = start_epoch + 1
    best_acc = evaluate(model, val_loader, device)
    print(f"Initial Validation Acc (from Stage 1): {best_acc:.2f}%")
    
    if os.path.exists(checkpoint_path):
        print(f"\n>>> Đang khôi phục huấn luyện từ {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f">>> Tiếp tục huấn luyện từ Epoch {current_epoch} (Best Acc: {best_acc:.2f}%)\n")

    # Training Loop (Stage 2)
    for epoch in range(current_epoch, end_epoch + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        start_time = time.time()
        
        for images, targets, indices in train_loader:
            images, targets = images.to(device), targets.to(device)
            batch_weights = sample_weights[indices].to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            
            # Weighted Loss calculation
            loss = criterion(logits, targets) 
            weighted_loss = (loss * batch_weights).mean()
            
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += weighted_loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        train_acc = 100. * correct / total
        print(f"Stage 2 - Epoch {epoch}/{end_epoch} | Time: {time.time()-start_time:.1f}s | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        
        val_acc = evaluate(model, val_loader, device)
        print(f"Validation Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('./submission', exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print("-> Đã lưu best model checkpoint mới cho Stage 2!")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, checkpoint_path)

if __name__ == '__main__':
    main()
