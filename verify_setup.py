"""Quick verification script for setup"""
import sys
import torch

print("=" * 60)
print("Environment Verification")
print("=" * 60)

print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"NCCL available: {torch.cuda.nccl.is_available()}")
    if torch.cuda.nccl.is_available():
        print(f"NCCL version: {torch.cuda.nccl.version()}")

print("\n" + "=" * 60)
print("Testing imports...")
print("=" * 60)

try:
    from config import get_config
    print("✓ config.py")
except Exception as e:
    print(f"✗ config.py: {e}")

try:
    from model import resnet50
    print("✓ model.py")
except Exception as e:
    print(f"✗ model.py: {e}")

try:
    from data import get_dataloaders
    print("✓ data.py")
except Exception as e:
    print(f"✗ data.py: {e}")

try:
    from augmentation import mixup_cutmix, mixed_ce
    print("✓ augmentation.py")
except Exception as e:
    print(f"✗ augmentation.py: {e}")

try:
    from utils import set_seed, save_checkpoint, AverageMeter
    print("✓ utils.py")
except Exception as e:
    print(f"✗ utils.py: {e}")

print("\n" + "=" * 60)
print("Testing model creation...")
print("=" * 60)

try:
    model = resnet50(num_classes=1000)
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if torch.cuda.is_available():
        model = model.cuda()
        print("✓ Model moved to CUDA")

        dummy_input = torch.randn(2, 3, 224, 224).cuda()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful: {output.shape}")
except Exception as e:
    print(f"✗ Model test failed: {e}")

print("\n" + "=" * 60)
print("Setup verification complete!")
print("=" * 60)
