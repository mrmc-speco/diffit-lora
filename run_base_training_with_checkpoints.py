#!/usr/bin/env python3
"""
Base Model Training with Checkpoint Management

Train DiffiT base model with resumable checkpointing.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add diffit to path
sys.path.insert(0, str(Path(__file__).parent))

from diffit.training.base_checkpoint_callbacks import (
    BaseModelCheckpointCallback,
    ResumeBaseModelCallback,
    get_latest_checkpoint,
    get_best_checkpoint
)
from diffit.training.data import DiffiTDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import json


def list_available_runs(base_dir: str = "checkpoints/base_training"):
    """List all available training runs."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print("No previous runs found.")
        return
    
    runs = []
    for run_dir in sorted(base_path.glob("run_*")):
        metadata_file = run_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                runs.append({
                    'run_number': metadata.get('run_number'),
                    'experiment_name': metadata.get('experiment_name'),
                    'created_at': metadata.get('created_at'),
                    'checkpoints_count': len(metadata.get('checkpoints', [])),
                    'best_checkpoint': metadata.get('best_checkpoint'),
                    'path': str(run_dir)
                })
    
    if not runs:
        print("No previous runs found.")
        return
    
    print("\n📋 Available Base Model Training Runs:")
    print("=" * 90)
    print(f"{'Run #':<8} {'Experiment':<25} {'Created':<20} {'Checkpoints':<12} {'Best Loss':<10}")
    print("-" * 90)
    
    for run in runs:
        best_loss = "N/A"
        if run['best_checkpoint'] and 'metrics' in run['best_checkpoint']:
            metrics = run['best_checkpoint']['metrics']
            if 'train_loss' in metrics:
                best_loss = f"{metrics['train_loss']:.4f}"
            elif 'val_loss' in metrics:
                best_loss = f"{metrics['val_loss']:.4f}"
        
        print(f"{run['run_number']:<8} "
              f"{run['experiment_name']:<25} "
              f"{run['created_at'][:19]:<20} "
              f"{run['checkpoints_count']:<12} "
              f"{best_loss:<10}")
    
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Train DiffiT Base Model with Checkpoint Management")
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training/base_training.yaml',
        help='Path to training configuration file'
    )
    
    parser.add_argument(
        '--run-number',
        type=int,
        default=None,
        help='Specific run number (creates new run if not specified)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint of specified run'
    )
    
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='List all available runs and exit'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs to train (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Training batch size (overrides config)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset to use: CIFAR, CIFAR100, IMAGENETTE (overrides config)'
    )
    
    args = parser.parse_args()
    
    # List runs and exit if requested
    if args.list_runs:
        list_available_runs()
        return
    
    print("🚀" * 40)
    print("   DiffiT Base Model Training with Checkpoint Management")
    print("🚀" * 40)
    
    # Load configuration
    print(f"\n📋 Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config.get('training', {})
    checkpoint_config = training_config.get('checkpoint', {})
    
    # Override config with command-line arguments
    if args.epochs is not None:
        training_config['num_epochs'] = args.epochs
    if args.batch_size is not None:
        training_config['batch_size_train'] = args.batch_size
    if args.learning_rate is not None:
        training_config['learning_rate'] = args.learning_rate
    if args.dataset is not None:
        training_config['dataset'] = args.dataset
    
    # Setup checkpoint callback
    checkpoint_callback = BaseModelCheckpointCallback(
        base_dir=checkpoint_config.get('base_dir', 'checkpoints/base_training'),
        run_number=args.run_number,
        experiment_name=training_config.get('experiment_name', 'diffit_base'),
        monitor=checkpoint_config.get('monitor', 'train_loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_every_n_epochs=checkpoint_config.get('save_every_n_epochs', 1),
        keep_last_n=checkpoint_config.get('keep_last_n', 3),
        verbose=True
    )
    
    callbacks = [checkpoint_callback]
    
    # Setup resume callback if resuming
    if args.resume:
        run_number = args.run_number if args.run_number is not None else checkpoint_callback.run_number
        checkpoint_path = get_latest_checkpoint(
            checkpoint_config.get('base_dir', 'checkpoints/base_training'),
            run_number
        )
        
        if checkpoint_path:
            resume_callback = ResumeBaseModelCallback(
                checkpoint_path=checkpoint_path,
                load_optimizer=True,
                load_scheduler=True,
                verbose=True
            )
            callbacks.append(resume_callback)
        else:
            print(f"\n⚠️  No checkpoint found for run {run_number}")
            print("   Starting new training...")
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Load dataset
    print(f"\n📊 Dataset Configuration:")
    print(f"   Dataset: {training_config.get('dataset')}")
    
    # Load appropriate dataset config
    dataset_name = training_config.get('dataset', 'CIFAR')
    if dataset_name == 'CIFAR':
        data_config_path = 'configs/data/cifar10.yaml'
    elif dataset_name == 'CIFAR100':
        data_config_path = 'configs/data/cifar100.yaml'
    elif dataset_name == 'IMAGENETTE':
        data_config_path = 'configs/data/imagenette.yaml'
    else:
        data_config_path = f'configs/data/{dataset_name.lower()}.yaml'
    
    print(f"   Config: {data_config_path}")
    
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Override batch size if specified
    if args.batch_size is not None:
        data_config['data']['batch_size_train'] = args.batch_size
    
    data_module = DiffiTDataModule(data_config)
    
    print(f"   Batch Size: {data_config['data'].get('batch_size_train', 64)}")
    
    # Training configuration summary
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {training_config.get('num_epochs', 3)}")
    print(f"   Learning Rate: {training_config.get('learning_rate', 0.001)}")
    print(f"   Optimizer: {training_config.get('optimizer', 'Adam')}")
    print(f"   Gradient Clip: {training_config.get('gradient_clip_val', 1.0)}")
    
    print(f"\n💾 Checkpoint Configuration:")
    print(f"   Run Number: {checkpoint_callback.run_number}")
    print(f"   Run Directory: {checkpoint_callback.run_dir}")
    print(f"   Save Every: {checkpoint_config.get('save_every_n_epochs', 1)} epoch(s)")
    print(f"   Keep Last: {checkpoint_config.get('keep_last_n', 3)} checkpoint(s)")
    print(f"   Monitor: {checkpoint_config.get('monitor', 'train_loss')} ({checkpoint_config.get('mode', 'min')})")
    
    print(f"\n{'='*80}")
    print("Ready to train!")
    print(f"{'='*80}\n")
    
    # TODO: Initialize your model here
    print("⚠️  Model initialization required!")
    print("\nTo complete the training setup, you need to:")
    print("1. Initialize your DiffiT model")
    print("2. Uncomment the trainer.fit() line below")
    print("\nExample:")
    print("  from diffit.models import DiffiTUNet  # Your model class")
    print("  model = DiffiTUNet(")
    print("      d_model=128,")
    print("      num_heads=2,")
    print("      img_size=32,")
    print("      learning_rate=training_config.get('learning_rate', 0.001)")
    print("  )")
    
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
    
    # Uncomment this when your model is ready:
    # trainer.fit(model, datamodule=data_module)
    
    print(f"\n✅ Configuration complete!")
    print(f"\n💡 To resume this run later, use:")
    print(f"   python {sys.argv[0]} --run-number {checkpoint_callback.run_number} --resume")
    
    print(f"\n📊 To see all runs, use:")
    print(f"   python {sys.argv[0]} --list-runs")


def show_usage_examples():
    """Show usage examples."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Base Model Training with Checkpoint Management - Usage Examples            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. Start new training run:                                                  ║
║     python run_base_training_with_checkpoints.py                             ║
║                                                                              ║
║  2. Resume from specific run:                                                ║
║     python run_base_training_with_checkpoints.py --run-number 1 --resume    ║
║                                                                              ║
║  3. List all available runs:                                                 ║
║     python run_base_training_with_checkpoints.py --list-runs                 ║
║                                                                              ║
║  4. Start new run with custom parameters:                                    ║
║     python run_base_training_with_checkpoints.py \\
║       --epochs 10 --batch-size 128 --learning-rate 1e-3                      ║
║                                                                              ║
║  5. Train on different dataset:                                              ║
║     python run_base_training_with_checkpoints.py --dataset CIFAR100          ║
║                                                                              ║
║  6. Resume run 2 with more epochs:                                           ║
║     python run_base_training_with_checkpoints.py \\
║       --run-number 2 --resume --epochs 20                                    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Checkpoint Structure:                                                       ║
║                                                                              ║
║  checkpoints/base_training/                                                  ║
║  ├── run_001/                                                                ║
║  │   ├── metadata.json             # Run metadata                            ║
║  │   ├── checkpoints/                                                        ║
║  │   │   ├── epoch_001.ckpt        # Full model per epoch                   ║
║  │   │   ├── epoch_002.ckpt                                                  ║
║  │   │   ├── best.ckpt              # Best checkpoint                        ║
║  │   │   └── last.ckpt              # Last checkpoint (for resume)           ║
║  │   └── logs/                      # TensorBoard logs                       ║
║  ├── run_002/                                                                ║
║  │   └── ...                                                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_usage_examples()
    
    main()
