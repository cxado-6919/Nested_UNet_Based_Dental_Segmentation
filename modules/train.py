import torch
import torch.optim as optim
from modules.losses import CombinedLoss, iou_score
from modules.utils import EarlyStopping

def train_model(model, train_loader, val_loader, num_epochs, device, checkpoint_path="best_model_checkpoint.pth"):
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=15, verbose=True, delta=0.001)

    best_iou = 0.0
    train_losses = []
    val_losses = []
    num_classes = 32  # 예시값

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)  
            # deep supervision이 켜진 경우 -> outputs가 list
            if isinstance(outputs, (list, tuple)):
                # 분기별로 loss 계산 후 합산/평균
                losses = [criterion(out, masks) for out in outputs]
                loss = sum(losses) / len(losses)
            else:
                # single output인 경우
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ----------------- Validation -----------------
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        count = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                
                # Validation 시에도 분기별 loss 계산
                if isinstance(outputs, (list, tuple)):
                    losses = [criterion(out, masks) for out in outputs]
                    loss = sum(losses) / len(losses)
                    # IoU는 마지막 분기를 기준으로 할지, 평균 낼지 사용자 결정
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

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val IoU: {avg_val_iou:.4f}")

        # 가장 높은 IoU 달성 시 체크포인트 저장
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), checkpoint_path)
            print("Best model saved!")
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    return model
