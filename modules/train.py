import torch
import torch.optim as optim
from modules.losses import iou_score, CombinedLoss
from modules.utils import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, num_epochs, device, checkpoint_path="best_model_checkpoint.pth"):
    optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-6)
    criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.0005)

    best_iou = 0.0
    train_losses = []
    val_losses = []
    num_classes = 32
    early_stopping_start_epoch = 15

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # deep supervision 사용 시 여러 출력에 대해 손실 계산 후 평균
            if isinstance(outputs, (list, tuple)):
                loss = sum([criterion(out, masks) for out in outputs]) / len(outputs)
            else:
                loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        count = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    loss = sum([criterion(out, masks) for out in outputs]) / len(outputs)
                    preds = torch.argmax(outputs[-1], dim=1)
                else:
                    loss = criterion(outputs, masks)
                    preds = torch.argmax(outputs, dim=1)
                val_loss += loss.item()
                batch_iou = iou_score(preds, masks, num_classes)
                total_iou += batch_iou
                count += 1
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = total_iou / count
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")
        
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), "best_model_checkpoint.pth")
            print("Best model saved!")
        
        scheduler.step(avg_val_loss)
        if epoch >= early_stopping_start_epoch:
            early_stopping(-avg_val_iou)  # IoU 최대화
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break


    # 학습 loss 그래프 시각화
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.show()

    return model