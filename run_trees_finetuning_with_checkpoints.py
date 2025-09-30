#!/usr/bin/env python3
"""
Trees Fine-tuning with Distributed Checkpoint Management

This script demonstrates:
1. Distributed checkpointing (separate LoRA and base model)
2. Run numbering system
3. Resume training from specific run
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add diffit to path
sys.path.insert(0, str(Path(__file__).parent))

from diffit.utils.checkpoint_manager import CheckpointManager, list_all_runs
from diffit.training.data import DiffiTDataModule
from diffit.training.checkpoint_callbacks import (
    DistributedCheckpointCallback,
    ResumeFromCheckpointCallback
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor


def list_available_runs():
    """List all available training runs."""
    runs = list_all_runs()
    
    if not runs:
        print("No previous runs found.")
        return
    
    print("\n📋 Available Training Runs:")
    print("=" * 80)
    print(f"{'Run #':<8} {'Experiment':<25} {'Created':<20} {'Checkpoints':<12}")
    print("-" * 80)
    
    for run in runs:
        print(f"{run['run_number']:<8} "
              f"{run['experiment_name']:<25} "
              f"{run['created_at'][:19]:<20} "
              f"{run['checkpoints_count']:<12}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DiffiT on Trees with Checkpoint Management")
    
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
        '--base-model',
        type=str,
        default='checkpoints/base/base_model_last.ckpt',
        help='Path to base model checkpoint'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs to train'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-4,
        help='Learning rate'
    )
    
    args = parser.parse_args()
    
    # List runs and exit if requested
    if args.list_runs:
        list_available_runs()
        return
    
    print("🌳" * 40)
    print("   DiffiT Trees Fine-tuning with Checkpoint Management")
    print("🌳" * 40)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        base_dir="checkpoints",
        experiment_name="diffit_lora_trees",
        run_number=args.run_number
    )
    
    print(checkpoint_manager.get_run_summary())
    
    # Load trees configuration
    print("\n📋 Loading trees dataset configuration...")
    with open('configs/data/cifar100_trees_only.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Create data module
    data_module = DiffiTDataModule(data_config)
    
    print("✅ Dataset configured: CIFAR-100 Trees Only")
    print(f"   Classes: maple_tree, oak_tree, palm_tree, pine_tree, willow_tree")
    
    # TODO: Initialize your model here
    # For demonstration, we'll show the setup
    print("\n🔧 Model Setup:")
    print(f"   Base Model: {args.base_model}")
    print(f"   Training Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.learning_rate}")
    
    # Example model initialization (you'll need to replace this with your actual model)
    """
    # Load base model
    if not args.resume:
        from diffit.models import DiffiTModel  # Your model class
        model = DiffiTModel.load_from_checkpoint(args.base_model)
        
        # Inject LoRA
        from diffit.lora import inject_lora
        lora_config = {...}  # Load from configs/lora/blockwise_config.yaml
        inject_lora(model, lora_config)
    else:
        # Load model with LoRA from checkpoint
        model = load_model_with_lora(checkpoint_manager)
    """
    
    # Setup callbacks
    callbacks = []
    
    # Distributed checkpoint callback
    checkpoint_callback = DistributedCheckpointCallback(
        checkpoint_manager=checkpoint_manager,
        monitor='val_loss',
        mode='min',
        save_every_n_epochs=1,
        save_base_model=not args.resume,  # Save base model only on new run
        keep_last_n=5,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Resume callback if resuming
    if args.resume:
        resume_callback = ResumeFromCheckpointCallback(
            checkpoint_manager=checkpoint_manager,
            load_optimizer=True,
            load_scheduler=True,
            verbose=True
        )
        callbacks.append(resume_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Save configs
    checkpoint_manager.save_config(
        training_config={
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'dataset': 'CIFAR-100 Trees Only',
            'base_model': args.base_model
        },
        lora_config={
            'enabled': True,
            'rank': 8,  # Example
            'alpha': 16   # Example
        }
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator='auto',
        devices='auto',
        precision='32-true',
        log_every_n_steps=25,
        default_root_dir=str(checkpoint_manager.logs_dir)
    )
    
    print("\n🚀 Training Configuration Complete!")
    print("\n📊 Checkpoint Structure:")
    print(f"   Run Directory: {checkpoint_manager.run_dir}")
    print(f"   Base Model: {checkpoint_manager.base_model_dir}/base_model.ckpt")
    print(f"   LoRA Weights: {checkpoint_manager.lora_weights_dir}/")
    print(f"   Logs: {checkpoint_manager.logs_dir}/")
    
    print("\n" + "="*80)
    print("To actually run training, you need to:")
    print("1. Initialize your DiffiT model")
    print("2. Inject LoRA adapters")
    print("3. Uncomment the trainer.fit() line below")
    print("="*80)
    
    # Uncomment this when your model is ready:
    # trainer.fit(model, datamodule=data_module)
    
    print("\n✅ Setup complete! See the checkpoint manager structure above.")
    print(f"\n💡 To resume this run later, use:")
    print(f"   python {sys.argv[0]} --run-number {checkpoint_manager.run_number} --resume")


def show_usage_examples():
    """Show usage examples."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Trees Fine-tuning with Checkpoint Management - Usage Examples              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. Start new training run:                                                  ║
║     python run_trees_finetuning_with_checkpoints.py                          ║
║                                                                              ║
║  2. Resume from specific run:                                                ║
║     python run_trees_finetuning_with_checkpoints.py --run-number 1 --resume ║
║                                                                              ║
║  3. List all available runs:                                                 ║
║     python run_trees_finetuning_with_checkpoints.py --list-runs             ║
║                                                                              ║
║  4. Start new run with custom parameters:                                    ║
║     python run_trees_finetuning_with_checkpoints.py \                        ║
║       --epochs 10 --batch-size 64 --learning-rate 1e-4                       ║
║                                                                              ║
║  5. Resume run 3 (automatically loads last checkpoint):                      ║
║     python run_trees_finetuning_with_checkpoints.py --run-number 3 --resume ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Checkpoint Structure:                                                       ║
║                                                                              ║
║  checkpoints/                                                                ║
║  ├── run_001/                                                                ║
║  │   ├── config.json              # Training & LoRA config                   ║
║  │   ├── metadata.json             # Run metadata                            ║
║  │   ├── base_model/                                                         ║
║  │   │   └── base_model.ckpt      # Base model weights (frozen)             ║
║  │   ├── lora_weights/                                                       ║
║  │   │   ├── epoch_001.ckpt       # LoRA weights per epoch                  ║
║  │   │   ├── epoch_002.ckpt                                                  ║
║  │   │   ├── best.ckpt             # Best checkpoint                         ║
║  │   │   └── last.ckpt             # Last checkpoint (for resume)            ║
║  │   └── logs/                     # TensorBoard logs                        ║
║  ├── run_002/                                                                ║
║  │   └── ...                                                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_usage_examples()
    
    main()
