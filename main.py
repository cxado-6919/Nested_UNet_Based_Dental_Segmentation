import os
import torch
from torch.utils.data import DataLoader, random_split
from modules.data import XRayDataset, get_train_transform, get_val_test_transform
from modules.train import train_model
from modules.utils import set_seed
from modules.visualize import visualize_prediction

# seed 설정 
set_seed(42)

# 데이터셋 경로 설정 
images_dir = 'your_path_here'
masks_dir = 'your_path_here'

# 데이터셋 생성 및 분할
full_dataset = XRayDataset(images_dir, masks_dir, transform=get_train_transform())
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# 검증/테스트 데이터셋에는 augmentation 없이 transform 적용
val_dataset.dataset.transform = get_val_test_transform()
test_dataset.dataset.transform = get_val_test_transform()

# DataLoader 구성
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 모델 생성 
from modules.NestedUNet import NestedUNet
num_classes = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NestedUNet(in_channels=3, out_channels=num_classes).to(device)

# 학습 실행
num_epochs = 200
checkpoint_path = "best_model_checkpoint.pth"
model = train_model(model, train_loader, val_loader, num_epochs, device, checkpoint_path=checkpoint_path)

# 테스트 평가
model.eval()
test_loss = 0.0
from modules.losses import CombinedLoss
criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        test_loss += loss.item()
avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")


mean = [0.3439, 0.3439, 0.3439]  
std = [0.2127, 0.2127, 0.2127]  

visualize_prediction(model, test_dataset, device, num_classes, mean, std)