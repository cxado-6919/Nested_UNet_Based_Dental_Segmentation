import torch
import numpy as np
import matplotlib.pyplot as plt

def unnormalize_image(img_tensor, mean, std):
    # 정규화된 이미지를 원래 스케일로 복원.

    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array(mean)
    std = np.array(std)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    return img_np

def visualize_prediction(model, dataset, device, num_classes, mean, std):
    # dataset에서 랜덤 샘플을 선택해서 원본 이미지, 실제 마스크, 예측 마스크, 그리고 오버레이 이미지를 시각화.
    import random
    model.eval()
    rand_idx = random.randint(0, len(dataset)-1)
    print(f"랜덤 테스트 샘플 인덱스: {rand_idx}")
    image, true_mask = dataset[rand_idx]
    image_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    pred_mask = torch.argmax(output, dim=1).squeeze(0)

    img_np = unnormalize_image(image, mean, std)
    true_mask_np = true_mask.cpu().numpy()
    pred_mask_np = pred_mask.cpu().numpy()

    # 3개의 서브플롯으로 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("원본 이미지")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask_np, cmap='jet', vmin=0, vmax=num_classes-1)
    plt.title("실제 마스크")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask_np, cmap='jet', vmin=0, vmax=num_classes-1)
    plt.title("예측 마스크")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

    # 오버레이 방식 시각화
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.imshow(pred_mask_np, cmap='jet', alpha=0.5, vmin=0, vmax=num_classes-1)
    plt.title("예측 마스크 오버레이")
    plt.axis("off")
    plt.show()