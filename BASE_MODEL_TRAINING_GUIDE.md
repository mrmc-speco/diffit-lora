# Base Model Training with Resume Capability

## Overview

Complete checkpoint management system for training DiffiT base models with:

- **🔢 Run Numbering**: Organize experiments with automatic run numbers
- **💾 Full Checkpointing**: Save complete model state at each epoch
- **♻️ Resume Training**: Continue from any checkpoint seamlessly
- **🗂️ Auto-cleanup**: Keep only recent checkpoints to save space
- **📊 Metadata Tracking**: Track all training metrics and progress

## Comparison: Base Model vs LoRA Fine-tuning Checkpoints

| Aspect | Base Model Training | LoRA Fine-tuning |
|--------|-------------------|------------------|
| **What's Saved** | Full model state | Base model (once) + LoRA weights |
| **Checkpoint Size** | ~500MB per epoch | Base: 500MB, LoRA: ~25MB per epoch |
| **Use Case** | Training from scratch | Adapting pre-trained model |
| **Disk Usage (10 epochs)** | ~5 GB | ~750 MB (85% savings) |
| **Resume Complexity** | Simple (one file) | Moderate (load base + LoRA) |

## Directory Structure

```
checkpoints/base_training/
├── run_001/                        ← First training run
│   ├── metadata.json               ← Run metadata and history
│   ├── checkpoints/
│   │   ├── epoch_001.ckpt         ← Full model state (500MB each)
│   │   ├── epoch_002.ckpt
│   │   ├── epoch_003.ckpt
│   │   ├── best.ckpt               ← Best checkpoint
│   │   └── last.ckpt               ← Last checkpoint (for resume)
│   └── logs/                       ← TensorBoard logs
├── run_002/                        ← Second training run
│   └── ...
└── run_003/                        ← Third training run
    └── ...
```

## Quick Start

### 1. Start New Training

```bash
python run_base_training_with_checkpoints.py
```

This will:
- Create `run_001` directory
- Train for configured epochs
- Save checkpoints automatically
- Track best checkpoint

### 2. Resume Training

```bash
# Resume from last checkpoint of run 1
python run_base_training_with_checkpoints.py --run-number 1 --resume
```

This will:
- Load model state from `last.ckpt`
- Restore optimizer state
- Restore scheduler state
- Continue from the next epoch

### 3. List All Runs

```bash
python run_base_training_with_checkpoints.py --list-runs
```

Output:
```
📋 Available Base Model Training Runs:
==================================================================================
Run #    Experiment                Created               Checkpoints  Best Loss
----------------------------------------------------------------------------------
1        diffit_base              2025-01-15 10:30:00   5            0.1234
2        diffit_base              2025-01-16 14:20:00   8            0.1156
3        diffit_base              2025-01-17 09:15:00   3            0.1289
==================================================================================
```

## Usage Examples

### Custom Parameters

```bash
# Train for 10 epochs with larger batch size
python run_base_training_with_checkpoints.py \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 1e-3
```

### Different Dataset

```bash
# Train on CIFAR-100 instead
python run_base_training_with_checkpoints.py --dataset CIFAR100

# Train on Imagenette
python run_base_training_with_checkpoints.py --dataset IMAGENETTE
```

### Resume with Modified Parameters

```bash
# Resume run 2 but train for more epochs
python run_base_training_with_checkpoints.py \
  --run-number 2 \
  --resume \
  --epochs 20
```

## Python API

### Basic Training

```python
from diffit.training.base_checkpoint_callbacks import BaseModelCheckpointCallback
import pytorch_lightning as pl

# Create checkpoint callback
checkpoint_callback = BaseModelCheckpointCallback(
    base_dir="checkpoints/base_training",
    experiment_name="my_experiment",
    monitor='train_loss',
    mode='min',
    save_every_n_epochs=1,
    keep_last_n=3
)

# Initialize your model
# model = YourDiffiTModel(...)

# Train
trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[checkpoint_callback]
)
trainer.fit(model, datamodule=data_module)

# Checkpoint callback automatically:
# - Creates run_XXX directory
# - Saves checkpoints every epoch
# - Tracks best checkpoint
# - Cleans up old checkpoints
```

### Resume Training

```python
from diffit.training.base_checkpoint_callbacks import (
    BaseModelCheckpointCallback,
    ResumeBaseModelCallback,
    get_latest_checkpoint
)

# Setup checkpoint callback for existing run
checkpoint_callback = BaseModelCheckpointCallback(
    base_dir="checkpoints/base_training",
    run_number=1,  # Specify run to continue
    save_every_n_epochs=1
)

# Get latest checkpoint path
checkpoint_path = get_latest_checkpoint("checkpoints/base_training", run_number=1)

# Setup resume callback
resume_callback = ResumeBaseModelCallback(
    checkpoint_path=checkpoint_path,
    load_optimizer=True,
    load_scheduler=True
)

# Train with both callbacks
trainer = pl.Trainer(
    max_epochs=20,
    callbacks=[checkpoint_callback, resume_callback]
)
trainer.fit(model, datamodule=data_module)
```

### Load Checkpoint Manually

```python
import torch

# Load best checkpoint
checkpoint = torch.load("checkpoints/base_training/run_001/checkpoints/best.ckpt")

# Extract components
model_state = checkpoint['state_dict']
optimizer_state = checkpoint['optimizer_state_dict']
scheduler_state = checkpoint['scheduler_state_dict']
epoch = checkpoint['epoch']
metrics = checkpoint['metrics']

print(f"Loaded checkpoint from epoch {epoch}")
print(f"Metrics: {metrics}")

# Load into model
model.load_state_dict(model_state)
```

## Configuration

### In `configs/training/base_training.yaml`

```yaml
training:
  # Checkpoint Management for Base Model Training
  checkpoint:
    enabled: true
    base_dir: "checkpoints/base_training"
    run_number: null  # Auto-increment if null
    save_every_n_epochs: 1
    keep_last_n: 3  # Keep last 3 checkpoints
    monitor: "train_loss"
    mode: "min"
    resume_from_run: null  # Run number to resume from
    resume_checkpoint: null  # Specific checkpoint path
```

### Checkpoint Settings Explained

- **`enabled`**: Turn checkpoint management on/off
- **`base_dir`**: Base directory for all runs
- **`run_number`**: 
  - `null` = auto-increment
  - `1, 2, 3...` = specific run number
- **`save_every_n_epochs`**: Checkpoint frequency
  - `1` = save every epoch
  - `2` = save every 2 epochs
- **`keep_last_n`**: Number of recent checkpoints to keep
  - Best checkpoint is always preserved
  - Older checkpoints are deleted
- **`monitor`**: Metric to track for best checkpoint
  - `train_loss`, `val_loss`, etc.
- **`mode`**: 
  - `min` = lower is better
  - `max` = higher is better

## Training Workflow

### Scenario 1: Fresh Training

```bash
# Day 1: Start training for 5 epochs
$ python run_base_training_with_checkpoints.py --epochs 5

Base Model Training - Run 001
Experiment: diffit_base
Run Directory: checkpoints/base_training/run_001

Epoch 1/5: train_loss=0.456
💾 Saved checkpoint: epoch_001.ckpt
   ⭐ New best checkpoint! train_loss=0.456

Epoch 2/5: train_loss=0.234
💾 Saved checkpoint: epoch_002.ckpt
   ⭐ New best checkpoint! train_loss=0.234

Epoch 3/5: train_loss=0.189
💾 Saved checkpoint: epoch_003.ckpt
   ⭐ New best checkpoint! train_loss=0.189

Epoch 4/5: train_loss=0.176
💾 Saved checkpoint: epoch_004.ckpt
   ⭐ New best checkpoint! train_loss=0.176

Epoch 5/5: train_loss=0.181
💾 Saved checkpoint: epoch_005.ckpt

🎉 Training completed!
⭐ Best checkpoint: Epoch 4, train_loss=0.176
```

### Scenario 2: Interrupted Training

```bash
# Day 1: Training starts
$ python run_base_training_with_checkpoints.py --epochs 10

Epoch 1/10: train_loss=0.456
Epoch 2/10: train_loss=0.234
Epoch 3/10: [POWER OUTAGE - Training stopped!]

# Day 2: Resume training
$ python run_base_training_with_checkpoints.py --run-number 1 --resume

📥 Resuming from checkpoint: checkpoints/base_training/run_001/checkpoints/last.ckpt
✅ Resumed from epoch 3
   Previous metrics: {'train_loss': 0.234}

Epoch 4/10: train_loss=0.189
Epoch 5/10: train_loss=0.176
...continues training...
```

### Scenario 3: Experiment Comparison

```bash
# Experiment 1: Standard learning rate
$ python run_base_training_with_checkpoints.py --learning-rate 1e-3
# Creates run_001

# Experiment 2: Lower learning rate
$ python run_base_training_with_checkpoints.py --learning-rate 1e-4
# Creates run_002

# Experiment 3: Higher learning rate
$ python run_base_training_with_checkpoints.py --learning-rate 1e-2
# Creates run_003

# Compare results
$ python run_base_training_with_checkpoints.py --list-runs

Run #    Experiment     Created            Checkpoints  Best Loss
1        diffit_base    2025-01-15 10:00   5            0.1234  ← Standard LR
2        diffit_base    2025-01-15 11:00   5            0.1156  ← Lower LR (best!)
3        diffit_base    2025-01-15 12:00   5            0.1890  ← Higher LR
```

## Checkpoint Contents

Each checkpoint file contains:

```python
checkpoint = {
    'epoch': 5,                          # Current epoch
    'state_dict': {...},                 # Full model weights
    'optimizer_state_dict': {...},      # Optimizer state
    'scheduler_state_dict': {...},      # LR scheduler state
    'metrics': {                         # Training metrics
        'train_loss': 0.176,
        'val_loss': 0.189,
        ...
    },
    'saved_at': '2025-01-15T10:30:00'   # Timestamp
}
```

## Metadata File

`metadata.json` tracks all run information:

```json
{
  "run_number": 1,
  "experiment_name": "diffit_base",
  "created_at": "2025-01-15T10:00:00",
  "checkpoints": [
    {
      "epoch": 1,
      "path": "checkpoints/base_training/run_001/checkpoints/epoch_001.ckpt",
      "metrics": {"train_loss": 0.456},
      "saved_at": "2025-01-15T10:15:00"
    },
    {
      "epoch": 2,
      "path": "checkpoints/base_training/run_001/checkpoints/epoch_002.ckpt",
      "metrics": {"train_loss": 0.234},
      "saved_at": "2025-01-15T10:30:00"
    }
  ],
  "best_checkpoint": {
    "epoch": 4,
    "path": "checkpoints/base_training/run_001/checkpoints/best.ckpt",
    "metrics": {"train_loss": 0.176}
  }
}
```

## Advanced Usage

### Custom Checkpoint Callback

```python
from diffit.training.base_checkpoint_callbacks import BaseModelCheckpointCallback

class MyCustomCallback(BaseModelCheckpointCallback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Custom logic before saving
        print("Custom pre-save logic...")
        
        # Call parent to save checkpoint
        super().on_train_epoch_end(trainer, pl_module)
        
        # Custom logic after saving
        print("Custom post-save logic...")
```

### Selective Loading

```python
import torch

checkpoint = torch.load("checkpoints/base_training/run_001/checkpoints/best.ckpt")

# Load only model weights (no optimizer/scheduler)
model.load_state_dict(checkpoint['state_dict'])

# For inference, you don't need optimizer/scheduler states
```

### Checkpoint Cleanup

```python
from pathlib import Path
import json

# Manual cleanup of very old runs
run_dir = Path("checkpoints/base_training/run_001")
metadata_file = run_dir / "metadata.json"

with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Keep only last 2 checkpoints
checkpoints = sorted(metadata['checkpoints'], key=lambda x: x['epoch'])
for ckpt in checkpoints[:-2]:
    ckpt_path = Path(ckpt['path'])
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"Deleted: {ckpt_path}")
```

## Integration with LoRA Fine-tuning

After base model training completes, use the best checkpoint for LoRA fine-tuning:

```bash
# 1. Train base model
python run_base_training_with_checkpoints.py --epochs 10
# Creates run_001

# 2. Copy best checkpoint to LoRA training directory
cp checkpoints/base_training/run_001/checkpoints/best.ckpt \
   checkpoints/base/base_model_last.ckpt

# 3. Start LoRA fine-tuning on trees
python run_trees_finetuning_with_checkpoints.py
```

Or programmatically:

```python
import shutil
from diffit.training.base_checkpoint_callbacks import get_best_checkpoint

# Get best base model checkpoint
best_ckpt = get_best_checkpoint("checkpoints/base_training", run_number=1)

# Copy to LoRA training location
shutil.copy(best_ckpt, "checkpoints/base/base_model_last.ckpt")

# Now start LoRA fine-tuning
```

## Best Practices

1. **Monitor appropriate metrics**:
   - Use `train_loss` if no validation set
   - Use `val_loss` if you have validation data
   
2. **Set reasonable `keep_last_n`**:
   - For experimentation: `keep_last_n=5-10`
   - For production: `keep_last_n=2-3`
   - Disk space: Each checkpoint ~500MB
   
3. **Regular cleanup**:
   - Delete old runs you don't need
   - Archive important runs to external storage
   
4. **Naming conventions**:
   - Use descriptive experiment names
   - Document hyperparameters in run notes
   
5. **Resume best practices**:
   - Always resume with `--resume` flag
   - Verify epoch count before resuming
   - Check that metrics make sense after resume

## Troubleshooting

### "Checkpoint not found"

```bash
$ python run_base_training_with_checkpoints.py --run-number 1 --resume
⚠️  No checkpoint found for run 1
   Starting new training...
```

**Solution**: Check if run exists:
```bash
ls checkpoints/base_training/run_001/checkpoints/
```

### "Out of disk space"

**Solution**: Clean up old checkpoints:
```bash
# Keep only last checkpoint
python -c "
from pathlib import Path
import shutil

# Delete old runs
for i in range(1, 10):
    run_dir = Path(f'checkpoints/base_training/run_{i:03d}')
    if run_dir.exists():
        shutil.rmtree(run_dir)
        print(f'Deleted run_{i:03d}')
"
```

### "Training is slow"

**Solution**: Adjust checkpoint frequency:
```yaml
# In config
checkpoint:
  save_every_n_epochs: 5  # Save every 5 epochs instead of every epoch
```

## Summary

**Base Model Training Checkpoints:**
- ✅ Complete model state saved
- ✅ Easy to resume training
- ✅ Track best models automatically
- ✅ Organize experiments with run numbers
- ✅ Automatic cleanup of old checkpoints
- ✅ Full metadata tracking

**Perfect for:**
- Training models from scratch
- Long training runs that might be interrupted
- Comparing different hyperparameters
- Keeping track of training progress

**Next Steps:**
1. Start training: `python run_base_training_with_checkpoints.py`
2. Monitor with TensorBoard: `tensorboard --logdir checkpoints/base_training/run_001/logs`
3. Resume if needed: `python run_base_training_with_checkpoints.py --run-number 1 --resume`
4. Use best checkpoint for LoRA fine-tuning!
