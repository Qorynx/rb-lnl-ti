import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from rb_lnl_ti import rb_lnl_ti
except ImportError:
    from upgrade_models.rb_lnl_ti import rb_lnl_ti

def build_clean_transforms():
    """
    Stage 4 uses clean, natural distribution with very minimal augmentation
    to optimize peak accuracy.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)), # Very light augmentation
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
    train_dataset = datasets.GTSRB(root=data_dir, split='train', transform=train_transform, download=True)
    val_dataset = datasets.GTSRB(root=data_dir, split='test', transform=val_transform, download=True)
    return train_dataset, val_dataset

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def unfreeze_all(model):
    """
    Unfreeze all layers for clean calibration.
    """
    for param in model.parameters():
        param.requires_grad = True

def main():
    # Stage 4 config
    start_epoch = 125 
    end_epoch = 139
    epochs_stage4 = end_epoch - start_epoch + 1
    
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = './data'
    base_model_path = './submission/rb_lnl_ti_gtsrb_stage3.pth' # Output from Stage 3
    checkpoint_path = './submission/latest_checkpoint_stage4.pth'
    best_model_path = './submission/rb_lnl_ti_gtsrb_final.pth' # Final model
    
    print(f"Using device: {device}")
    
    if not os.path.exists(base_model_path):
        print(f"ERROR: Cannot find Stage 3 model {base_model_path}. Please run Stage 3 first!")
        return

    train_transform, val_transform = build_clean_transforms()
    train_dataset, val_dataset = get_datasets(data_dir, train_transform, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model Setup
    model = rb_lnl_ti(num_classes=43, pretrained=False)
    
    print(f"Loading Stage 3 model weights from {base_model_path}...")
    model.load_state_dict(torch.load(base_model_path, map_location='cpu'))
    
    # M6: Keep Residual Head active
    model.use_residual = True
    print("Residual Head is active.")
    
    # Unfreeze all blocks to settle representation on clean data
    unfreeze_all(model)
    model.to(device)
    
    # M6: Very low learning rate and reduced label smoothing
    # We use a base LR of 1e-5 going down to 1e-6
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01) # Reduced to 0.01
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage4, eta_min=1e-6)
    
    current_epoch = start_epoch + 1
    best_acc = evaluate(model, val_loader, device)
    print(f"Initial Validation Acc (from Stage 3): {best_acc:.2f}%")
    
    if os.path.exists(checkpoint_path):
        print(f"\n>>> Đang khôi phục huấn luyện từ {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f">>> Tiếp tục huấn luyện từ Epoch {current_epoch} (Best Acc: {best_acc:.2f}%)\n")

    # Training Loop (Stage 4)
    for epoch in range(current_epoch, end_epoch + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        start_time = time.time()
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            
            # Pure CE loss on clean distribution
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
        print(f"Stage 4 - Epoch {epoch}/{end_epoch} | Time: {time.time()-start_time:.1f}s | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        
        val_acc = evaluate(model, val_loader, device)
        print(f"Validation Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('./submission', exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print("-> Đã lưu FINAL best model checkpoint!")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, checkpoint_path)

if __name__ == '__main__':
    main()
