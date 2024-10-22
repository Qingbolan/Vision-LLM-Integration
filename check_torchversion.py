import torchvision
print(f"Torchvision version: {torchvision.__version__}")

from torchvision.models import vit_b_16, ViT_B_16_Weights
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
print("\nModel structure:")
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")