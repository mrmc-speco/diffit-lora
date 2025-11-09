# Distributed Checkpoint Management System

## Overview

The DiffiT LoRA checkpoint management system provides:

- **🔢 Run Numbering**: Automatic run numbering for organized experiments
- **📦 Distributed Storage**: Separate LoRA and base model checkpoints
- **♻️ Resume Training**: Easy resume from any run number
- **🗂️ Auto-cleanup**: Keep only recent checkpoints to save space
- **📊 Metadata Tracking**: Track all training metrics and configurations

## Directory Structure

```
checkpoints/
├── run_001/
│   ├── config.json              # Training & LoRA configurations
│   ├── metadata.json             # Run metadata and checkpoint history
│   ├── base_model/
│   │   └── base_model.ckpt      # Base model weights (frozen, saved once)
│   ├── lora_weights/
│   │   ├── epoch_001.ckpt       # LoRA weights + optimizer state per epoch
│   │   ├── epoch_002.ckpt
│   │   ├── epoch_003.ckpt
│   │   ├── best.ckpt             # Best checkpoint based on monitored metric
│   │   └── last.ckpt             # Last checkpoint (for resuming)
│   └── logs/                     # TensorBoard logs
├── run_002/
│   └── ...
└── run_003/
    └── ...
```

## Key Features

### 1. Distributed Checkpoints

**Base Model** (saved once):
- Contains only non-LoRA parameters
- Frozen during training
- Shared across epochs
- Location: `run_XXX/base_model/base_model.ckpt`

**LoRA Weights** (saved per epoch):
- Contains only LoRA adapter parameters (`lora_A`, `lora_B`)
- Includes optimizer state for resuming
- Includes scheduler state
- Includes training metrics
- Location: `run_XXX/lora_weights/epoch_XXX.ckpt`

### 2. Run Numbering

Runs are automatically numbered sequentially:
- `run_001` → First training run
- `run_002` → Second training run
- `run_003` → Third training run
- etc.

You can also specify a run number manually for organization.

### 3. Checkpoint Types

Each run maintains multiple checkpoint types:

- **Epoch Checkpoints**: `epoch_001.ckpt`, `epoch_002.ckpt`, etc.
  - Full training state at each epoch
  - Used for resuming or analyzing training progression

- **Best Checkpoint**: `best.ckpt`
  - Copy of the epoch with best monitored metric (e.g., lowest val_loss)
  - Automatically updated when a better checkpoint is found

- **Last Checkpoint**: `last.ckpt`
  - Most recent checkpoint
  - Used for resuming interrupted training

### 4. Metadata Tracking

`metadata.json` contains:
```json
{
  "run_number": 1,
  "experiment_name": "diffit_lora_trees",
  "created_at": "2025-01-15T10:30:00",
  "checkpoints": [
    {
      "epoch": 1,
      "path": "run_001/lora_weights/epoch_001.ckpt",
      "metrics": {"val_loss": 0.234, "train_loss": 0.189},
      "saved_at": "2025-01-15T11:00:00"
    }
  ],
  "best_checkpoint": {
    "epoch": 3,
    "path": "run_001/lora_weights/best.ckpt",
    "metrics": {"val_loss": 0.198}
  },
  "training_config": {...},
  "lora_config": {...}
}
```

## Usage

### Python API

#### Starting a New Training Run

```python
from diffit.utils.checkpoint_manager import CheckpointManager
from diffit.training.checkpoint_callbacks import DistributedCheckpointCallback

# Create checkpoint manager (auto-increments run number)
checkpoint_manager = CheckpointManager(
    base_dir="checkpoints",
    experiment_name="diffit_lora_trees"
)

# Use in PyTorch Lightning
checkpoint_callback = DistributedCheckpointCallback(
    checkpoint_manager=checkpoint_manager,
    monitor='val_loss',
    mode='min',
    save_every_n_epochs=1,
    keep_last_n=5
)

trainer = pl.Trainer(callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=data_module)
```

#### Resuming from a Specific Run

```python
from diffit.utils.checkpoint_manager import CheckpointManager
from diffit.training.checkpoint_callbacks import (
    DistributedCheckpointCallback,
    ResumeFromCheckpointCallback
)

# Load existing run
checkpoint_manager = CheckpointManager(
    base_dir="checkpoints",
    experiment_name="diffit_lora_trees",
    run_number=3  # Resume run 3
)

# Setup callbacks
resume_callback = ResumeFromCheckpointCallback(
    checkpoint_manager=checkpoint_manager,
    load_optimizer=True,
    load_scheduler=True
)

checkpoint_callback = DistributedCheckpointCallback(
    checkpoint_manager=checkpoint_manager,
    save_base_model=False  # Don't save base model again
)

trainer = pl.Trainer(callbacks=[resume_callback, checkpoint_callback])
trainer.fit(model, datamodule=data_module)
```

#### Loading Checkpoints Manually

```python
# Load base model
base_checkpoint = checkpoint_manager.load_base_model()
model.load_state_dict(base_checkpoint['state_dict'], strict=False)

# Load best LoRA weights
lora_checkpoint = checkpoint_manager.load_lora_weights(load_best=True)
model.load_state_dict(lora_checkpoint['lora_state_dict'], strict=False)

# Load specific epoch
lora_checkpoint = checkpoint_manager.load_lora_weights(epoch=5)

# Load last checkpoint for resuming
resume_dict = checkpoint_manager.load_for_resume()
```

#### Listing Available Runs

```python
from diffit.utils.checkpoint_manager import list_all_runs

runs = list_all_runs("checkpoints")
for run in runs:
    print(f"Run {run['run_number']}: {run['experiment_name']}")
    print(f"  Created: {run['created_at']}")
    print(f"  Checkpoints: {run['checkpoints_count']}")
```

### Command-Line Usage

#### Start New Training

```bash
# Start new run (auto-increments to run_001)
python run_trees_finetuning_with_checkpoints.py

# With custom parameters
python run_trees_finetuning_with_checkpoints.py \
  --epochs 10 \
  --batch-size 64 \
  --learning-rate 1e-4
```

#### Resume Training

```bash
# Resume from last checkpoint of run 3
python run_trees_finetuning_with_checkpoints.py \
  --run-number 3 \
  --resume

# Resume with different parameters
python run_trees_finetuning_with_checkpoints.py \
  --run-number 3 \
  --resume \
  --epochs 15 \
  --learning-rate 1e-5
```

#### List All Runs

```bash
python run_trees_finetuning_with_checkpoints.py --list-runs
```

Output:
```
📋 Available Training Runs:
================================================================================
Run #    Experiment                Created               Checkpoints
--------------------------------------------------------------------------------
1        diffit_lora_trees         2025-01-15 10:30:00  5
2        diffit_lora_trees         2025-01-16 14:20:00  8
3        diffit_lora_trees         2025-01-17 09:15:00  3
================================================================================
```

## Configuration

In `configs/training/lora_finetuning.yaml`:

```yaml
training:
  checkpoint:
    enabled: true
    base_dir: "checkpoints"
    run_number: null  # Auto-increment if null
    save_every_n_epochs: 1
    keep_last_n: 5  # Keep last 5 checkpoints
    monitor: "val_loss"
    mode: "min"
    save_base_model: true
    resume_from_run: null  # Run number to resume from
```

## Benefits

### 1. Disk Space Efficiency

- **Base model saved once**: No duplication across epochs
- **Only LoRA weights per epoch**: Typically 1-5% of full model size
- **Auto-cleanup**: Keeps only recent checkpoints

Example savings:
- Full model: 500 MB
- LoRA weights: 25 MB (5% of full model)
- 10 epochs traditional: 5 GB
- 10 epochs distributed: 500 MB + (25 MB × 10) = 750 MB
- **Savings: 4.25 GB (85%)**

### 2. Easy Experimentation

- Track multiple training runs
- Compare different hyperparameters
- Resume from any point
- Organized by run number

### 3. Safe Resuming

- Automatically loads:
  - LoRA weights
  - Optimizer state
  - Scheduler state
  - Training metrics
- Continue exactly where you left off

### 4. Model Deployment

```python
# For deployment, merge LoRA into base model
checkpoint_manager = CheckpointManager(run_number=3)

# Load base
base_ckpt = checkpoint_manager.load_base_model()
model.load_state_dict(base_ckpt['state_dict'])

# Load and fuse best LoRA
lora_ckpt = checkpoint_manager.load_lora_weights(load_best=True)
model.load_state_dict(lora_ckpt['lora_state_dict'], strict=False)

# Fuse LoRA for inference
from diffit.lora import fuse_lora_weights
fuse_lora_weights(model)

# Save merged model
torch.save(model.state_dict(), 'production_model.ckpt')
```

## Advanced Usage

### Custom Checkpoint Manager

```python
class MyCheckpointManager(CheckpointManager):
    def save_lora_weights(self, *args, **kwargs):
        # Custom logic before saving
        print("Saving with custom logic...")
        path = super().save_lora_weights(*args, **kwargs)
        # Custom logic after saving
        return path
```

### Checkpoint Cleanup Policy

```python
# Keep last 10 checkpoints
checkpoint_manager.cleanup_old_checkpoints(keep_last_n=10)

# Keep last 3, always preserve best
checkpoint_manager.cleanup_old_checkpoints(
    keep_last_n=3,
    keep_best=True
)
```

### Export Run Summary

```python
summary = checkpoint_manager.get_run_summary()
print(summary)

# Get structured data
best_info = checkpoint_manager.get_best_checkpoint_info()
all_checkpoints = checkpoint_manager.list_checkpoints()
```

## Troubleshooting

### Run Number Conflicts

If you manually specify a run number that exists:
```python
# This will load the existing run
checkpoint_manager = CheckpointManager(run_number=3)

# To force a new run with specific number, delete the old one first
import shutil
shutil.rmtree("checkpoints/run_003")
```

### Missing Base Model

If resuming and base model is missing:
```python
checkpoint_callback = DistributedCheckpointCallback(
    checkpoint_manager=checkpoint_manager,
    save_base_model=True  # Save it during this run
)
```

### Corrupted Checkpoint

Load from a different epoch:
```python
# Try different epochs
for epoch in [5, 4, 3, 2, 1]:
    try:
        ckpt = checkpoint_manager.load_lora_weights(epoch=epoch)
        break
    except:
        continue
```

## Best Practices

1. **Save base model once** at the start of training
2. **Use meaningful experiment names** for organization
3. **Set appropriate `keep_last_n`** based on disk space
4. **Always monitor a metric** for best checkpoint selection
5. **Use run numbers** for experiment tracking
6. **Clean up old runs** periodically to save disk space
7. **Export metadata** for experiment tracking systems

## Integration with Existing Systems

### With TensorBoard

```python
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(
    save_dir=checkpoint_manager.logs_dir,
    name=checkpoint_manager.experiment_name
)
```

### With Weights & Biases

```python
import wandb

wandb.init(
    project="diffit-lora",
    name=f"run_{checkpoint_manager.run_number:03d}",
    config=checkpoint_manager.metadata['training_config']
)
```

### With MLflow

```python
import mlflow

mlflow.start_run(run_name=f"run_{checkpoint_manager.run_number:03d}")
mlflow.log_params(checkpoint_manager.metadata['training_config'])
```
