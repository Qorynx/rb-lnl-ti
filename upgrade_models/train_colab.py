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

def get_datasets(data_dir, train_transform, val_transform):
    os.makedirs(data_dir, exist_ok=True)
    # Use the indexed wrapper for training so we can map weights in Stage 2
    train_dataset = IndexedGTSRB(root=data_dir, split='train', transform=train_transform, download=True)
    # Validation doesn't strictly need index mapping, but we can keep consistency
    val_dataset = IndexedGTSRB(root=data_dir, split='test', transform=val_transform, download=True)
    return train_dataset, val_dataset

def evaluate_and_log_difficulty(model, dataloader, device, epoch, save_dir):
    model.eval()
    all_data = []
    confusion_matrix = torch.zeros(43, 43)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    print("Evaluating and logging difficulty...")
    with torch.no_grad():
        for i, (images, targets, indices) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            loss = criterion(logits, targets)
            
            top2_prob, top2_idx = probs.topk(2, dim=1)
            top1_prob, top2_prob = top2_prob[:, 0], top2_prob[:, 1]
            predicted_label = top2_idx[:, 0]
            margin = top1_prob - top2_prob
            confidence = probs[torch.arange(probs.size(0)), targets]
            
            for t, p in zip(targets.view(-1), predicted_label.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
            for j in range(images.size(0)):
                is_correct = (predicted_label[j] == targets[j]).item()
                l, m, c = loss[j].item(), margin[j].item(), confidence[j].item()
                idx = indices[j].item()
                
                if not is_correct or l > 1.0: group = "Hard"
                elif is_correct and m < 0.5: group = "Ambiguous"
                else: group = "Easy"
                
                all_data.append({
                    'image_id': idx,
                    'true_label': targets[j].item(), 'predicted_label': predicted_label[j].item(),
                    'loss': l, 'confidence': c, 'top1_prob': top1_prob[j].item(),
                    'top2_prob': top2_prob[j].item(), 'margin': m, 'difficulty_group': group
                })

    df = pd.DataFrame(all_data)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f'sample_difficulty_epoch_{epoch}.csv'), index=False)
    pd.DataFrame(confusion_matrix.numpy()).to_csv(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.csv'), index=False)
    print(f"Saved difficulty log and confusion matrix for epoch {epoch}.")
    return (df['true_label'] == df['predicted_label']).mean() * 100

def main():
    epochs = 90
    batch_size = 32
    learning_rate = 3e-4
    weight_decay = 0.04
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = './data'
    save_dir = './submission/results'
    
    print(f"Using device: {device}")
    
    train_transform, val_transform = build_transforms()
    train_dataset, val_dataset = get_datasets(data_dir, train_transform, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = rb_lnl_ti(num_classes=43, pretrained=False)
    model.use_residual = False 
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Resume Training Logic
    checkpoint_path = './submission/latest_checkpoint.pth'
    best_model_path = './submission/rb_lnl_ti_gtsrb.pth'
    start_epoch = 1
    best_acc = 0.0
    
    if os.path.exists(checkpoint_path):
        print(f"\n>>> Đang khôi phục huấn luyện từ {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f">>> Tiếp tục huấn luyện từ Epoch {start_epoch} (Best Acc hiện tại: {best_acc:.2f}%)\n")
    else:
        print("\n>>> Bắt đầu huấn luyện từ đầu.\n")

    # Training Loop (Stage 1)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        start_time = time.time()
        
        for images, targets, indices in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        train_acc = 100. * correct / total
        print(f"Epoch {epoch}/{epochs} | Time: {time.time()-start_time:.1f}s | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        
        if epoch % 10 == 0 or epoch == epochs:
            val_acc = evaluate_and_log_difficulty(model, val_loader, device, epoch, save_dir)
            print(f"Validation Acc: {val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs('./submission', exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print("Saved new best model checkpoint.")
                
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, checkpoint_path)

if __name__ == '__main__':
    main()
