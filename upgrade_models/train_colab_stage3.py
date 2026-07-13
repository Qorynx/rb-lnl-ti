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

def get_confusion_pairs(cm_csv_path, threshold=2):
    """
    Read confusion matrix and return a dict of true_class -> list(confused_classes)
    """
    print(f"Loading confusion matrix from {cm_csv_path}...")
    cm_df = pd.read_csv(cm_csv_path)
    cm = cm_df.values
    np.fill_diagonal(cm, 0) # ignore correct predictions
    
    confused_dict = {i: [] for i in range(43)}
    total_pairs = 0
    for i in range(43):
        for j in range(43):
            if cm[i, j] > threshold:
                confused_dict[i].append(j)
                total_pairs += 1
                
    print(f"Found {total_pairs} confused pairs with freq > {threshold}.")
    return confused_dict

def pairwise_margin_loss(logits, targets, confused_dict, margin=0.2):
    """
    Pairwise margin loss for confusion correction
    L_pair = max(0, margin - z_y + z_j)
    """
    loss = 0.0
    count = 0
    for i, target in enumerate(targets):
        t = target.item()
        confused_classes = confused_dict[t]
        for conf_cls in confused_classes:
            z_y = logits[i, t]
            z_j = logits[i, conf_cls]
            loss += F.relu(margin - z_y + z_j)
            count += 1
            
    if count > 0:
        return loss / count
    return torch.tensor(0.0, device=logits.device, requires_grad=True)

def setup_differential_lr(model):
    """
    Freeze early blocks, apply different LRs to late blocks, base head, and residual head.
    """
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    late_params = []
    base_head_params = []
    residual_head_params = []
    
    for name, param in model.named_parameters():
        if 'residual_head' in name or 'residual_scale' in name:
            param.requires_grad = True
            residual_head_params.append(param)
        elif 'backbone.head' in name:
            param.requires_grad = True
            base_head_params.append(param)
        elif 'backbone.blocks' in name:
            block_idx = int(name.split('.')[2])
            if block_idx >= 8: # Late blocks (8, 9, 10, 11)
                param.requires_grad = True
                late_params.append(param)
        elif 'backbone.norm' in name:
            param.requires_grad = True
            late_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': late_params, 'lr': 1e-5},
        {'params': base_head_params, 'lr': 5e-5},
        {'params': residual_head_params, 'lr': 2e-4}
    ], weight_decay=0.04)
    
    return optimizer

def main():
    # Stage 3 config
    start_epoch = 110 
    end_epoch = 124
    epochs_stage3 = end_epoch - start_epoch + 1
    
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = './data'
    cm_csv_path = './submission/results/confusion_matrix_epoch_90.csv' # Output from Stage 1
    base_model_path = './submission/rb_lnl_ti_gtsrb_stage2.pth' # Output from Stage 2
    checkpoint_path = './submission/latest_checkpoint_stage3.pth'
    best_model_path = './submission/rb_lnl_ti_gtsrb_stage3.pth'
    
    print(f"Using device: {device}")
    
    if not os.path.exists(cm_csv_path):
        print(f"ERROR: Cannot find {cm_csv_path}. Please run Stage 1 first!")
        return
        
    if not os.path.exists(base_model_path):
        print(f"ERROR: Cannot find Stage 2 model {base_model_path}. Please run Stage 2 first!")
        return

    train_transform, val_transform = build_transforms()
    train_dataset, val_dataset = get_datasets(data_dir, train_transform, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Prepare Confusion Pairs
    confused_dict = get_confusion_pairs(cm_csv_path, threshold=2)
    lambda_pair = 0.05
    margin = 0.2
    
    # Model Setup
    model = rb_lnl_ti(num_classes=43, pretrained=False)
    
    print(f"Loading Stage 2 model weights from {base_model_path}...")
    # strict=False because residual head might not have been fully optimized/saved properly if it wasn't used, 
    # but it was initialized. Actually, it's safe to load with strict=True since the parameters exist.
    model.load_state_dict(torch.load(base_model_path, map_location='cpu'))
    
    # M4: Bật Residual Head
    model.use_residual = True
    print("Activated Residual Head!")
    
    model.to(device)
    
    # Setup Optimizer with Differential LR (M4)
    optimizer = setup_differential_lr(model)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.03)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage3)
    
    current_epoch = start_epoch + 1
    best_acc = evaluate(model, val_loader, device)
    print(f"Initial Validation Acc (from Stage 2): {best_acc:.2f}%")
    
    if os.path.exists(checkpoint_path):
        print(f"\n>>> Đang khôi phục huấn luyện từ {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f">>> Tiếp tục huấn luyện từ Epoch {current_epoch} (Best Acc: {best_acc:.2f}%)\n")

    # Training Loop (Stage 3)
    for epoch in range(current_epoch, end_epoch + 1):
        model.train()
        total_loss, total_ce, total_pair = 0, 0, 0
        correct, total = 0, 0
        start_time = time.time()
        
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            
            # M5: Tổng hợp Loss
            loss_ce = criterion_ce(logits, targets)
            loss_pair = pairwise_margin_loss(logits, targets, confused_dict, margin=margin)
            
            loss = loss_ce + lambda_pair * loss_pair
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_pair += loss_pair.item()
            
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        train_acc = 100. * correct / total
        print(f"Stage 3 - Epoch {epoch}/{end_epoch} | Time: {time.time()-start_time:.1f}s | "
              f"Loss: {total_loss/len(train_loader):.4f} (CE: {total_ce/len(train_loader):.4f}, "
              f"Pair: {total_pair/len(train_loader):.4f}) | Train Acc: {train_acc:.2f}%")
        
        val_acc = evaluate(model, val_loader, device)
        print(f"Validation Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('./submission', exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print("-> Đã lưu best model checkpoint mới cho Stage 3!")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, checkpoint_path)

if __name__ == '__main__':
    main()
