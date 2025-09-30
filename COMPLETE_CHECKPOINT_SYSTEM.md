# Complete Checkpoint Management System

## Overview

You now have a **complete checkpoint management system** for both base model training and LoRA fine-tuning with resume capability!

## 🎯 Two Types of Checkpointing

### 1. Base Model Training
- **Full model checkpoints**
- **Use case**: Training from scratch
- **Size**: ~500MB per checkpoint
- **Resume**: Load complete model state
- **Script**: `run_base_training_with_checkpoints.py`

### 2. LoRA Fine-tuning  
- **Distributed checkpoints** (base + LoRA separate)
- **Use case**: Adapting pre-trained models
- **Size**: Base 500MB (once) + LoRA 25MB per epoch
- **Resume**: Load base model + LoRA weights
- **Script**: `run_trees_finetuning_with_checkpoints.py`

## 📁 Complete Directory Structure

```
checkpoints/
├── base_training/                  ← Base model training
│   ├── run_001/
│   │   ├── metadata.json
│   │   ├── checkpoints/
│   │   │   ├── epoch_001.ckpt     ← Full model (500MB)
│   │   │   ├── epoch_002.ckpt
│   │   │   ├── best.ckpt
│   │   │   └── last.ckpt           ← For resume
│   │   └── logs/
│   └── run_002/
│       └── ...
│
├── run_001/                        ← LoRA fine-tuning (trees)
│   ├── config.json
│   ├── metadata.json
│   ├── base_model/
│   │   └── base_model.ckpt        ← Base model (500MB, saved once)
│   ├── lora_weights/
│   │   ├── epoch_001.ckpt          ← LoRA only (25MB)
│   │   ├── epoch_002.ckpt
│   │   ├── best.ckpt
│   │   └── last.ckpt               ← For resume
│   └── logs/
│
└── run_002/                        ← Another LoRA run
    └── ...
```

## 🚀 Quick Start Guide

### Step 1: Train Base Model

```bash
# Start new base model training
python run_base_training_with_checkpoints.py

# Creates: checkpoints/base_training/run_001/
# Trains for 3 epochs (default)
# Saves checkpoints automatically
```

If training gets interrupted:
```bash
# Resume from where you left off
python run_base_training_with_checkpoints.py --run-number 1 --resume
```

### Step 2: Fine-tune with LoRA (Trees)

```bash
# Make sure you have a base model checkpoint first
# Then start LoRA fine-tuning on trees dataset
python run_trees_finetuning_with_checkpoints.py

# Creates: checkpoints/run_001/
# Separates base model and LoRA weights
# Trains for 5 epochs (default)
```

If interrupted:
```bash
# Resume LoRA fine-tuning
python run_trees_finetuning_with_checkpoints.py --run-number 1 --resume
```

### Step 3: List All Experiments

```bash
# List base model training runs
python run_base_training_with_checkpoints.py --list-runs

# List LoRA fine-tuning runs  
python run_trees_finetuning_with_checkpoints.py --list-runs
```

## 📊 Feature Comparison

| Feature | Base Model | LoRA Fine-tuning |
|---------|-----------|------------------|
| **Checkpoint Type** | Full model | Distributed (base + LoRA) |
| **Size per Epoch** | ~500MB | ~25MB (LoRA only) |
| **Disk Usage (10 epochs)** | ~5 GB | ~750 MB |
| **Savings** | - | 85% less space |
| **Resume Complexity** | Simple | Moderate |
| **Best For** | Training from scratch | Domain adaptation |
| **Training Speed** | Standard | Faster (fewer params) |

## 🔄 Complete Workflow Example

### Scenario: Train base model, then fine-tune on trees

```bash
# ============================================
# DAY 1: Train Base Model on CIFAR-10
# ============================================

# Start training
$ python run_base_training_with_checkpoints.py --epochs 10

Base Model Training - Run 001
Epoch 1/10: train_loss=0.456
Epoch 2/10: train_loss=0.234
Epoch 3/10: [CRASH!]

# Resume training
$ python run_base_training_with_checkpoints.py --run-number 1 --resume

📥 Resuming from epoch 3...
Epoch 4/10: train_loss=0.189
...
Epoch 10/10: train_loss=0.095

🎉 Training completed!
⭐ Best checkpoint: Epoch 9, train_loss=0.092

# Copy best checkpoint for LoRA use
$ cp checkpoints/base_training/run_001/checkpoints/best.ckpt \
     checkpoints/base/base_model_last.ckpt

# ============================================
# DAY 2: Fine-tune on Trees with LoRA
# ============================================

# Start LoRA fine-tuning
$ python run_trees_finetuning_with_checkpoints.py --epochs 5

LoRA Fine-tuning - Run 001
🌳 Dataset: Trees only (5 classes)
💾 Saving base model (500MB)...
💾 Saving LoRA weights (epoch 1, 25MB)...
💾 Saving LoRA weights (epoch 2, 25MB)...
[POWER OUTAGE!]

# Resume LoRA fine-tuning
$ python run_trees_finetuning_with_checkpoints.py --run-number 1 --resume

📥 Resuming from epoch 2...
✅ Loaded base model
✅ Loaded LoRA weights
Epoch 3/5: val_loss=0.145
Epoch 4/5: val_loss=0.128  ← Best!
Epoch 5/5: val_loss=0.132

🎉 Fine-tuning completed!
⭐ Best checkpoint: Epoch 4, val_loss=0.128

# ============================================
# DAY 3: Compare Multiple Experiments
# ============================================

# Experiment 1: Trees with default params
$ python run_trees_finetuning_with_checkpoints.py
# Creates run_001

# Experiment 2: Trees with higher LR
$ python run_trees_finetuning_with_checkpoints.py --learning-rate 1e-3
# Creates run_002

# Experiment 3: Animals instead of trees
$ vim configs/training/lora_finetuning.yaml
# Change to: data_config: "configs/data/cifar100_animals_only.yaml"
$ python run_trees_finetuning_with_checkpoints.py
# Creates run_003

# Compare all experiments
$ python run_trees_finetuning_with_checkpoints.py --list-runs

Run #    Experiment              Dataset      Checkpoints  Best Loss
1        diffit_lora_trees       Trees (5)    5            0.128  ← Best!
2        diffit_lora_trees       Trees (5)    5            0.145
3        diffit_lora_trees       Animals (40) 5            0.234
```

## 💡 Key Concepts

### Run Numbers
- Automatically increment: `run_001`, `run_002`, `run_003`...
- Each run is independent
- Easy to compare experiments
- Can specify manually with `--run-number`

### Checkpoint Types
1. **Epoch checkpoints**: `epoch_001.ckpt`, `epoch_002.ckpt`
2. **Best checkpoint**: `best.ckpt` (lowest/highest monitored metric)
3. **Last checkpoint**: `last.ckpt` (for resuming)

### Resume Capability
- **Base Model**: Loads full model state
- **LoRA**: Loads base model + LoRA weights separately
- Both restore: optimizer state, scheduler state, training metrics

### Auto-cleanup
- Keeps last N checkpoints (configurable)
- Always preserves best checkpoint
- Saves disk space automatically

## 📋 Command Reference

### Base Model Training

```bash
# Start new run
python run_base_training_with_checkpoints.py

# Start with custom params
python run_base_training_with_checkpoints.py \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --dataset CIFAR100

# Resume run 1
python run_base_training_with_checkpoints.py --run-number 1 --resume

# List all runs
python run_base_training_with_checkpoints.py --list-runs
```

### LoRA Fine-tuning

```bash
# Start new run (trees)
python run_trees_finetuning_with_checkpoints.py

# Start with custom params
python run_trees_finetuning_with_checkpoints.py \
  --epochs 10 \
  --batch-size 64 \
  --learning-rate 5e-4

# Resume run 1
python run_trees_finetuning_with_checkpoints.py --run-number 1 --resume

# List all runs
python run_trees_finetuning_with_checkpoints.py --list-runs
```

## 🔧 Configuration Files

### Base Model Training
- **Config**: `configs/training/base_training.yaml`
- **Dataset**: `configs/data/cifar10.yaml`, `cifar100.yaml`, etc.

### LoRA Fine-tuning
- **Config**: `configs/training/lora_finetuning.yaml`
- **Dataset**: `configs/data/cifar100_trees_only.yaml`
- **LoRA**: `configs/lora/blockwise_config.yaml`

## 📚 Documentation

1. **Base Model Training**: `BASE_MODEL_TRAINING_GUIDE.md`
   - Full guide for base model training
   - Resume examples
   - API reference

2. **LoRA Fine-tuning**: `QUICK_START_TREES_FINETUNING.md`
   - Trees fine-tuning guide
   - Distributed checkpoints
   - Class filtering

3. **Checkpoint System**: `CHECKPOINT_MANAGEMENT.md`
   - Complete checkpoint manager docs
   - Advanced usage
   - Best practices

4. **This Guide**: `COMPLETE_CHECKPOINT_SYSTEM.md`
   - Overview of entire system
   - Workflow examples
   - Quick reference

## 🎯 Use Cases

### Use Case 1: Long Training Run

**Problem**: Training takes 24 hours but might crash

**Solution**:
```bash
# Start training
python run_base_training_with_checkpoints.py --epochs 100

# If it crashes at epoch 47, resume:
python run_base_training_with_checkpoints.py --run-number 1 --resume

# Continues from epoch 48
```

### Use Case 2: Hyperparameter Search

**Problem**: Need to try different learning rates

**Solution**:
```bash
# Try different learning rates, each creates new run
for lr in 1e-2 1e-3 1e-4 1e-5; do
  python run_base_training_with_checkpoints.py --learning-rate $lr
done

# Compare results
python run_base_training_with_checkpoints.py --list-runs
```

### Use Case 3: Dataset Adaptation

**Problem**: Pre-trained on CIFAR-10, adapt to trees

**Solution**:
```bash
# Step 1: Train base on CIFAR-10
python run_base_training_with_checkpoints.py --dataset CIFAR

# Step 2: Copy best checkpoint
cp checkpoints/base_training/run_001/checkpoints/best.ckpt \
   checkpoints/base/base_model_last.ckpt

# Step 3: Fine-tune on trees with LoRA
python run_trees_finetuning_with_checkpoints.py
```

### Use Case 4: Resume After Crash

**Problem**: Power outage during fine-tuning

**Solution**:
```bash
# Before crash (epoch 3/10)
python run_trees_finetuning_with_checkpoints.py --epochs 10
# [CRASH at epoch 3]

# After reboot
python run_trees_finetuning_with_checkpoints.py --run-number 1 --resume
# Automatically continues from epoch 4
```

## ⚡ Performance Tips

1. **Save frequency**: Adjust `save_every_n_epochs` based on training speed
   - Fast training: `save_every_n_epochs=1`
   - Slow training: `save_every_n_epochs=5`

2. **Disk space**: Set `keep_last_n` appropriately
   - Lots of space: `keep_last_n=10`
   - Limited space: `keep_last_n=2-3`

3. **Monitor right metric**:
   - No validation set: `monitor=train_loss`
   - With validation: `monitor=val_loss`

## 🔒 Safety Features

- ✅ **Automatic backup**: Last checkpoint always saved
- ✅ **Best preservation**: Best checkpoint never deleted
- ✅ **Metadata tracking**: All runs tracked with timestamps
- ✅ **Safe resume**: Validates checkpoint before loading
- ✅ **Graceful failures**: Falls back to new training if resume fails

## 🎓 Learning Path

1. **Start Here**: Read this document (COMPLETE_CHECKPOINT_SYSTEM.md)
2. **Base Training**: Read BASE_MODEL_TRAINING_GUIDE.md
3. **LoRA Fine-tuning**: Read QUICK_START_TREES_FINETUNING.md
4. **Advanced**: Read CHECKPOINT_MANAGEMENT.md
5. **Practice**: Run the example scripts

## 📞 Quick Help

### "How do I start training?"
```bash
python run_base_training_with_checkpoints.py
```

### "How do I resume?"
```bash
python run_base_training_with_checkpoints.py --run-number 1 --resume
```

### "How do I see all my experiments?"
```bash
python run_base_training_with_checkpoints.py --list-runs
python run_trees_finetuning_with_checkpoints.py --list-runs
```

### "Where are my checkpoints?"
```bash
ls checkpoints/base_training/run_001/checkpoints/
ls checkpoints/run_001/lora_weights/
```

### "How do I use the best checkpoint?"
```python
import torch

# For base model
ckpt = torch.load("checkpoints/base_training/run_001/checkpoints/best.ckpt")
model.load_state_dict(ckpt['state_dict'])

# For LoRA
base_ckpt = torch.load("checkpoints/run_001/base_model/base_model.ckpt")
lora_ckpt = torch.load("checkpoints/run_001/lora_weights/best.ckpt")
model.load_state_dict(base_ckpt['state_dict'], strict=False)
model.load_state_dict(lora_ckpt['lora_state_dict'], strict=False)
```

## ✅ Summary

You now have:

**For Base Model Training:**
- ✅ Run-numbered checkpoints
- ✅ Full model state saving
- ✅ Resume from any epoch
- ✅ Auto-cleanup
- ✅ Best checkpoint tracking

**For LoRA Fine-tuning:**
- ✅ Distributed checkpoints (base + LoRA)
- ✅ 85% disk space savings
- ✅ Resume from any epoch
- ✅ Trees dataset filtering
- ✅ Multiple dataset support

**Both Systems:**
- ✅ PyTorch Lightning integration
- ✅ Metadata tracking
- ✅ TensorBoard logging
- ✅ Command-line interface
- ✅ Python API

**Happy Training! 🚀**
