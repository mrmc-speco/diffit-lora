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
    
    # Setup resume callback to load existing checkpoint
    resume_callback = ResumeBaseModelCallback(
        checkpoint_path=existing_checkpoint,
        load_optimizer=True,  # Load optimizer state if available
        load_scheduler=True,  # Load scheduler state if available
        verbose=True
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks = [checkpoint_callback, resume_callback, lr_monitor]
    
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
    
    # TODO: Initialize your model here
    from diffit.models.unet import UShapedNetwork
    model = UShapedNetwork(
        d_model=128,
        num_heads=2,
        img_size=32,
        learning_rate=training_config.get('learning_rate', 0.001)
    )
    
    # Uncomment this when your model is ready:
    trainer.fit(model, datamodule=data_module)
    
    print(f"\n💡 After training completes, new checkpoints will be in:")
    print(f"   {checkpoint_callback.run_dir}")
    print(f"\n✅ Setup complete!")


if __name__ == "__main__":
    main()
