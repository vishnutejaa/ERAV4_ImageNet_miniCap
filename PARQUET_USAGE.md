# Using Parquet Datasets with ImageNet Training

This repository now supports **both** traditional ImageFolder datasets and **Hugging Face parquet datasets**. This makes it easy to train on datasets from the Hugging Face Hub without manual conversion.

## Quick Start

### Option 1: Traditional ImageFolder Format (Default)

By default, the code expects the standard ImageNet directory structure:

```
data/
├── train/
│   ├── n01440764/
│   │   ├── image1.JPEG
│   │   └── image2.JPEG
│   └── n01443537/
│       └── ...
└── val/
    ├── n01440764/
    └── ...
```

No configuration changes needed - just run training as usual:

```bash
python train.py
```

### Option 2: Hugging Face Parquet Datasets

If you have parquet files from Hugging Face (e.g., ImageNet-1k downloaded from the Hub):

```
/data/imagenet-1k/data/
├── train-00001-of-00294.parquet
├── train-00002-of-00294.parquet
├── ...
├── validation-00001-of-00006.parquet
└── ...
```

#### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the `datasets` library needed for parquet support.

#### Step 2: Configure for Parquet

Modify your `config.py` or create a custom config:

```python
from config import TrainingConfig

# Create custom config for parquet datasets
cfg = TrainingConfig()
cfg.use_parquet = True
cfg.parquet_data_dir = "/data/imagenet-1k/data"
cfg.parquet_train_pattern = "train-*.parquet"
cfg.parquet_val_pattern = "validation-*.parquet"
```

Or directly modify the defaults in `config.py`:

```python
@dataclass
class TrainingConfig:
    # ... other settings ...

    # Parquet dataset support
    use_parquet: bool = True  # ✅ Enable parquet mode
    parquet_data_dir: str = "/data/imagenet-1k/data"  # ✅ Set your path
    parquet_train_pattern: str = "train-*.parquet"
    parquet_val_pattern: str = "validation-*.parquet"
```

#### Step 3: Run Training

```bash
python train.py
```

That's it! The code will automatically load from parquet files.

## How It Works

### ParquetDataset Class

The `ParquetDataset` class in `data.py` wraps Hugging Face's `datasets` library:

- **Lazy loading**: Images are loaded on-demand, not all at once
- **Memory efficient**: Uses streaming to avoid loading everything into RAM
- **Automatic handling**: Works with multiple parquet files matching a pattern
- **Transform support**: Full compatibility with torchvision transforms

### Dataset Structure

Parquet files should contain:
- `image`: PIL Image or image bytes
- `label`: Integer class label (0-999 for ImageNet-1k)

Example schema:
```python
{
  "image": Image(decode=True),
  "label": int
}
```

### Performance Considerations

**Parquet advantages:**
- No need to extract/convert datasets
- Works directly with Hugging Face Hub downloads
- Efficient storage (compressed)

**ImageFolder advantages:**
- Slightly faster I/O (direct file access)
- More familiar format

For most use cases, performance is comparable. Choose based on your workflow preference.

## Advanced Usage

### Custom Parquet Patterns

If your parquet files have different naming:

```python
cfg.parquet_train_pattern = "my_train_*.parquet"
cfg.parquet_val_pattern = "my_val_*.parquet"
```

### Switching Between Formats

To switch back to ImageFolder:

```python
cfg.use_parquet = False
cfg.train_dir = "data/train"
cfg.val_dir = "data/val"
```

### Distributed Training

Both formats fully support DDP (DistributedDataParallel):

```bash
torchrun --nproc_per_node=4 train.py
```

The `DistributedSampler` works seamlessly with both `ImageFolder` and `ParquetDataset`.

## Troubleshooting

### ImportError: No module named 'datasets'

Install the datasets library:
```bash
pip install datasets
```

### ValueError: parquet_data_dir must be specified

Make sure to set `parquet_data_dir` when `use_parquet=True`:
```python
cfg.parquet_data_dir = "/path/to/your/parquet/files"
```

### File pattern not matching

Check that your pattern correctly matches your files:
```bash
ls /data/imagenet-1k/data/train-*.parquet
```

Adjust `parquet_train_pattern` if needed.

## Example: Download and Train on ImageNet-1k from Hugging Face

```bash
# 1. Download ImageNet-1k (requires authentication)
huggingface-cli login
huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset --local-dir /data/imagenet-1k

# 2. Configure for parquet
# Edit config.py to set:
#   use_parquet = True
#   parquet_data_dir = "/data/imagenet-1k/data"

# 3. Train
python train.py
```

## Questions?

The implementation automatically detects whether to use parquet or ImageFolder based on the `use_parquet` flag in your config. Both modes support:

- ✅ Full torchvision transform pipelines
- ✅ RandAugment, MixUp, CutMix
- ✅ Distributed training (DDP)
- ✅ Multi-GPU training
- ✅ Efficient dataloading with num_workers

Choose the format that works best for your workflow!
