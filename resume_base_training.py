#!/usr/bin/env python3
"""
Resume Base Model Training from Existing Checkpoint

This script helps you resume training from your existing base model checkpoint
located at: checkpoints/base/base_model_last.ckpt
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from diffit.training.base_checkpoint_callbacks import (
    BaseModelCheckpointCallback,
    ResumeBaseModelCallback
)
from diffit.training.data import DiffiTDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor


def main():
    print("🔄 Resuming Base Model Training")
    print("=" * 60)
    
    # Your existing checkpoint
    existing_checkpoint = "checkpoints/base/base_model_last.ckpt"
    
    print(f"📥 Will resume from: {existing_checkpoint}")
    print(f"   File size: 53MB")
    
    # Load configuration
    with open('configs/training/base_training.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config.get('training', {})
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {training_config.get('num_epochs', 3)}")
    print(f"   Learning Rate: {training_config.get('learning_rate', 0.001)}")
    print(f"   Dataset: {training_config.get('dataset', 'CIFAR')}")
    
    # Setup checkpoint callback for future saves
    checkpoint_callback = BaseModelCheckpointCallback(
        base_dir="checkpoints/base_training",
        run_number=None,  # Will create new run
        experiment_name="diffit_base_resumed",
        monitor='train_loss',
        mode='min',
        save_every_n_epochs=1,
        keep_last_n=3,
        verbose=True
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks = [checkpoint_callback, lr_monitor]
    
    # Load dataset
    dataset_name = training_config.get('dataset', 'CIFAR')
    if dataset_name == 'CIFAR':
        data_config_path = 'configs/data/cifar10.yaml'
    elif dataset_name == 'CIFAR100':
        data_config_path = 'configs/data/cifar100.yaml'
    else:
        data_config_path = f'configs/data/{dataset_name.lower()}.yaml'
    
    print(f"\n📊 Loading dataset: {data_config_path}")
    
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    data_module = DiffiTDataModule(data_config)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=training_config.get('num_epochs', 3),
        callbacks=callbacks,
        accelerator=training_config.get('accelerator', 'auto'),
        devices=training_config.get('devices', 'auto'),
        precision=training_config.get('precision', '32-true'),
        gradient_clip_val=training_config.get('gradient_clip_val', 1.0),
        log_every_n_steps=training_config.get('log_every_n_steps', 25),
        default_root_dir=str(checkpoint_callback.logs_dir)
    )
    
    print("\n" + "="*60)
    print("Ready to resume training!")
    print("="*60)
    print("\n⚠️  To complete the setup, you need to:")
    print("1. Initialize your DiffiT model")
    print("2. Uncomment the trainer.fit() line below")
    print("\nExample:")
    print("  from diffit.models.unet import UShapedNetwork")
    print("  model = UShapedNetwork(")
    print("      d_model=128,")
    print("      num_heads=2,")
    print("      img_size=32,")
    print("      learning_rate=0.001")
    print("  )")
    print("\nThe ResumeCallback will automatically load the checkpoint")
    print("when training starts!")
    
    # Initialize your model with all required parameters
    from diffit.models.unet import UShapedNetwork
    import torch
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UShapedNetwork(
        learning_rate=training_config.get('learning_rate', 0.001),
        d_model=128,
        num_heads=2,
        dropout=0.1,
        d_ff=256,
        img_size=32,
        device=device,
        denoising_steps=500
    )
    
    print(f"\n✅ Model initialized:")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get starting epoch from checkpoint and fix version compatibility
    import torch
    import pytorch_lightning as pl
    
    checkpoint = torch.load(existing_checkpoint, map_location='cpu')
    starting_epoch = checkpoint.get('epoch', 0)
    checkpoint_callback.set_starting_epoch(starting_epoch)
    
    # Fix checkpoint compatibility by adding missing PyTorch Lightning version
    if 'pytorch-lightning_version' not in checkpoint:
        checkpoint['pytorch-lightning_version'] = pl.__version__
        # Save the fixed checkpoint to a temporary location
        fixed_checkpoint_path = "temp_fixed_checkpoint.ckpt"
        torch.save(checkpoint, fixed_checkpoint_path)
        existing_checkpoint = fixed_checkpoint_path
        print(f"🔧 Fixed checkpoint compatibility - saved to: {existing_checkpoint}")
    
    print(f"📊 Will continue from epoch {starting_epoch + 1}")
    
    # Start training with proper resume
    trainer.fit(model, datamodule=data_module, ckpt_path=existing_checkpoint)
    
    print(f"\n💡 After training completes, new checkpoints will be in:")
    print(f"   {checkpoint_callback.run_dir}")
    
    # Clean up temporary checkpoint file if it was created
    if existing_checkpoint == "temp_fixed_checkpoint.ckpt":
        import os
        if os.path.exists("temp_fixed_checkpoint.ckpt"):
            os.remove("temp_fixed_checkpoint.ckpt")
            print("🧹 Cleaned up temporary checkpoint file")
    
    print(f"\n✅ Setup complete!")


if __name__ == "__main__":
    main()
