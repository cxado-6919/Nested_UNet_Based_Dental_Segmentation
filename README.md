# Nested_UNet_Based_Dental_Segmentation
 Project to develop dental X-ray segmentation models using Nested U-Net architectures, aiming to accurately segment individual teeth from dental images.

## Code Execution Environment


The code was executed in a cloud computing environment configured with the following specifications:

- **Operating System**: Ubuntu Linux 5.15.0-97-generic (x86_64)
- **Processor (CPU)**: 16 vCore, x86_64 architecture
- **GPU**: NVIDIA A100 80GB PCIe
  - **Driver Version**: 535.183.06
  - **CUDA Version**: 12.2
  - **cuDNN Version**: 9.1.0
- **Memory (RAM)**: 192GB
- **Python Version**: 3.10.13 (GCC 11.3.0)

### Python Libraries

Key Python libraries used in training:

- `torch==2.6.0`
- `torchvision==0.21.0`
- `numpy==2.2.3`
- `matplotlib==3.10.1`

## Dataset

The model was trained using the dataset available at [Teeth Segmentation on Dental X-ray Images](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images).
