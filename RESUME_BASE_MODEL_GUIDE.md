# How to Resume Base Model Training

## Your Situation

You have an existing base model checkpoint:
- **Location**: `checkpoints/base/base_model_last.ckpt`
- **Size**: 53MB
- **Type**: PyTorch Lightning checkpoint

## 🎯 Three Ways to Resume Training

### Method 1: Using the Resume Script (Easiest)

I've created a script specifically for you:

```bash
python resume_base_training.py
```

**What it does:**
1. Loads your existing checkpoint: `checkpoints/base/base_model_last.ckpt`
2. Creates a new run directory for continued training
3. Automatically restores:
   - Model weights ✅
   - Optimizer state (if available) ✅
   - Scheduler state (if available) ✅
   - Current epoch ✅

**You need to add:**
- Your model initialization code (see example in the script)

### Method 2: Using PyTorch Lightning's Built-in Resume

PyTorch Lightning has native support for resuming:

```python
import pytorch_lightning as pl

# Your model initialization
# model = YourModel(...)

# Your data
# data_module = DiffiTDataModule(...)

# Create trainer with resume path
trainer = pl.Trainer(
    max_epochs=10,
    # Other settings...
)

# Resume training from checkpoint
trainer.fit(
    model, 
    datamodule=data_module,
    ckpt_path="checkpoints/base/base_model_last.ckpt"  # ← Resume from here
)
```

**Advantages:**
- Simple one-liner
- Automatic state restoration
- No callback needed

**Example script:**

```python
#!/usr/bin/env python3
import pytorch_lightning as pl
import yaml
from diffit.training.data import DiffiTDataModule
from diffit.models.unet import UShapedNetwork  # Your model

# Load config
with open('configs/training/base_training.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load dataset
with open('configs/data/cifar10.yaml', 'r') as f:
    data_config = yaml.safe_load(f)
data_module = DiffiTDataModule(data_config)

# Initialize model
model = UShapedNetwork(
    d_model=128,
    num_heads=2,
    img_size=32,
    learning_rate=0.001
)

# Create trainer
trainer = pl.Trainer(
    max_epochs=10,  # Continue for 10 more epochs
    accelerator='auto',
    devices='auto'
)

# Resume from checkpoint
trainer.fit(
    model,
    datamodule=data_module,
    ckpt_path="checkpoints/base/base_model_last.ckpt"  # Your checkpoint
)
```

### Method 3: Manual Checkpoint Loading

If you want full control:

```python
import torch
import pytorch_lightning as pl

# Load checkpoint manually
checkpoint = torch.load("checkpoints/base/base_model_last.ckpt")

# Inspect what's in it
print("Checkpoint contents:")
print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  Keys: {checkpoint.keys()}")

# Initialize model
# model = YourModel(...)

# Load only model weights
model.load_state_dict(checkpoint['state_dict'])

# Optionally load optimizer state
# optimizer.load_state_dict(checkpoint['optimizer_states'][0])

# Start training from this point
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model, datamodule=data_module)
```

## 📋 What's Inside Your Checkpoint?

Your `base_model_last.ckpt` file contains:

```python
checkpoint = {
    'epoch': X,                          # Last completed epoch
    'global_step': Y,                    # Total training steps
    'state_dict': {...},                 # Model weights
    'optimizer_states': [{...}],         # Optimizer state
    'lr_schedulers': [{...}],           # Learning rate scheduler state
    'callbacks': {...},                  # Callback states
    'pytorch-lightning_version': '...',  # PL version
    # ... and more metadata
}
```

## 🔍 Inspect Your Checkpoint

Before resuming, you might want to see what's in your checkpoint:

```python
import torch

# Load checkpoint
ckpt = torch.load("checkpoints/base/base_model_last.ckpt", map_location='cpu')

# Print information
print("="*60)
print("Checkpoint Information")
print("="*60)
print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"Global Step: {ckpt.get('global_step', 'N/A')}")
print(f"PyTorch Lightning Version: {ckpt.get('pytorch-lightning_version', 'N/A')}")

# Check if optimizer state exists
if 'optimizer_states' in ckpt:
    print(f"Optimizer States: {len(ckpt['optimizer_states'])} optimizer(s)")

# Check model parameters
if 'state_dict' in ckpt:
    num_params = len(ckpt['state_dict'])
    print(f"Model Parameters: {num_params} tensors")
    
    # Sample parameter names
    param_names = list(ckpt['state_dict'].keys())[:5]
    print(f"Sample parameter names:")
    for name in param_names:
        print(f"  - {name}")

print("="*60)
```

Save this as `inspect_checkpoint.py` and run:
```bash
python inspect_checkpoint.py
```

## 💡 Recommended Workflow

### If you want to continue training in a new organized run:

```bash
# Step 1: Run the resume script (adds checkpoint management)
python resume_base_training.py

# Step 2: Add your model initialization code to resume_base_training.py

# Step 3: Run it
python resume_base_training.py
```

**Benefits:**
- Creates new organized run directory
- Saves new checkpoints separately
- Preserves original checkpoint
- Tracks training progress

### If you want simple continuation:

```python
# Direct PyTorch Lightning resume
trainer.fit(model, datamodule, ckpt_path="checkpoints/base/base_model_last.ckpt")
```

**Benefits:**
- Simplest approach
- One line of code
- Automatic state restoration

## 🚀 Complete Example

Here's a complete working example:

```python
#!/usr/bin/env python3
"""
Complete example: Resume base model training
"""

import pytorch_lightning as pl
import yaml
from diffit.training.data import DiffiTDataModule
from diffit.training.base_checkpoint_callbacks import BaseModelCheckpointCallback
from pytorch_lightning.callbacks import LearningRateMonitor

# Load configurations
with open('configs/training/base_training.yaml', 'r') as f:
    train_config = yaml.safe_load(f)

with open('configs/data/cifar10.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

# Setup data
data_module = DiffiTDataModule(data_config)

# Initialize model (replace with your actual model)
from diffit.models.unet import UShapedNetwork
model = UShapedNetwork(
    d_model=128,
    num_heads=2,
    dropout=0.1,
    d_ff=256,
    img_size=32,
    device='cpu',  # or 'cuda'
    denoising_steps=500,
    learning_rate=0.001
)

# Setup new checkpoint callback for future saves
checkpoint_callback = BaseModelCheckpointCallback(
    base_dir="checkpoints/base_training",
    experiment_name="base_resumed",
    monitor='train_loss',
    mode='min'
)

# Setup callbacks
callbacks = [
    checkpoint_callback,
    LearningRateMonitor(logging_interval='step')
]

# Create trainer
trainer = pl.Trainer(
    max_epochs=10,  # Train for 10 more epochs
    callbacks=callbacks,
    accelerator='auto',
    devices='auto',
    gradient_clip_val=1.0,
    log_every_n_steps=25
)

# Resume training from your existing checkpoint
print("🔄 Resuming training from checkpoints/base/base_model_last.ckpt")
trainer.fit(
    model,
    datamodule=data_module,
    ckpt_path="checkpoints/base/base_model_last.ckpt"
)

print("✅ Training completed!")
print(f"New checkpoints saved to: {checkpoint_callback.run_dir}")
```

## 🎯 Key Points

1. **Your checkpoint is compatible** with PyTorch Lightning's resume functionality
2. **Simplest way**: Use `ckpt_path` parameter in `trainer.fit()`
3. **Organized way**: Use the resume script with checkpoint callbacks
4. **The checkpoint contains**: Model weights, optimizer state, scheduler state, epoch info

## ⚠️ Important Notes

1. **Model Architecture Must Match**: Your model initialization must match the architecture in the checkpoint
2. **Dependencies**: Make sure you have the same PyTorch Lightning version (or compatible)
3. **Hyperparameters**: You can change hyperparameters (learning rate, epochs, etc.) when resuming
4. **Epoch Count**: If checkpoint was at epoch 5, resuming with `max_epochs=10` will train epochs 6-10

## 🔧 Troubleshooting

### "RuntimeError: Error(s) in loading state_dict"

**Cause**: Model architecture doesn't match checkpoint

**Solution**: Check your model initialization parameters

### "Checkpoint not found"

**Cause**: Wrong path

**Solution**: Verify the path:
```bash
ls -lh checkpoints/base/base_model_last.ckpt
```

### "Out of memory"

**Cause**: GPU memory issue

**Solution**: Reduce batch size or use CPU:
```python
trainer = pl.Trainer(accelerator='cpu')
```

## 📊 Next Steps

1. **Choose your method** (I recommend Method 2 for simplicity)
2. **Add your model initialization**
3. **Run the training**
4. **Monitor progress** with TensorBoard:
   ```bash
   tensorboard --logdir checkpoints/base_training/
   ```

## 💬 Quick Commands

```bash
# Inspect checkpoint
python -c "import torch; ckpt=torch.load('checkpoints/base/base_model_last.ckpt'); print(f'Epoch: {ckpt.get(\"epoch\")}')"

# Resume with PyTorch Lightning (simplest)
# Add to your training script:
# trainer.fit(model, datamodule, ckpt_path="checkpoints/base/base_model_last.ckpt")

# Or use the resume script
python resume_base_training.py
```

Your checkpoint is ready to go! Just pick a method and start training! 🚀
