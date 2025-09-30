#!/usr/bin/env python3
"""
Inspect Base Model Checkpoint

Quick script to see what's inside checkpoints/base/base_model_last.ckpt
"""

import torch
from pathlib import Path

def inspect_checkpoint(checkpoint_path="checkpoints/base/base_model_last.ckpt"):
    """Inspect a PyTorch Lightning checkpoint."""
    
    print("🔍 Inspecting Checkpoint")
    print("=" * 70)
    
    # Check if file exists
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Get file size
    file_size = ckpt_file.stat().st_size / (1024 * 1024)  # MB
    print(f"📁 File: {checkpoint_path}")
    print(f"📊 Size: {file_size:.2f} MB")
    
    # Load checkpoint
    print("\n📥 Loading checkpoint...")
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return
    
    print("\n" + "=" * 70)
    print("📋 Checkpoint Contents")
    print("=" * 70)
    
    # Top-level keys
    print(f"\n🔑 Top-level keys:")
    for key in ckpt.keys():
        print(f"  • {key}")
    
    # Epoch information
    if 'epoch' in ckpt:
        print(f"\n⏱️  Training Progress:")
        print(f"  • Epoch: {ckpt['epoch']}")
        if 'global_step' in ckpt:
            print(f"  • Global Step: {ckpt['global_step']}")
    
    # PyTorch Lightning version
    if 'pytorch-lightning_version' in ckpt:
        print(f"\n⚡ PyTorch Lightning Version: {ckpt['pytorch-lightning_version']}")
    
    # Model state
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"\n🧠 Model State:")
        print(f"  • Total parameters: {len(state_dict)} tensors")
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"  • Total parameter count: {total_params:,}")
        print(f"  • Estimated model size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
        
        # Sample parameter names
        param_names = list(state_dict.keys())
        print(f"\n  📝 Sample parameter names (first 10):")
        for name in param_names[:10]:
            shape = tuple(state_dict[name].shape)
            print(f"     • {name}: {shape}")
        
        if len(param_names) > 10:
            print(f"     ... and {len(param_names) - 10} more")
    
    # Optimizer state
    if 'optimizer_states' in ckpt:
        opt_states = ckpt['optimizer_states']
        print(f"\n🎯 Optimizer State:")
        print(f"  • Number of optimizers: {len(opt_states)}")
        if opt_states:
            print(f"  • Optimizer type: {opt_states[0].get('optimizer_type', 'Unknown')}")
    
    # Learning rate schedulers
    if 'lr_schedulers' in ckpt:
        lr_schedulers = ckpt['lr_schedulers']
        print(f"\n📈 LR Schedulers:")
        print(f"  • Number of schedulers: {len(lr_schedulers)}")
    
    # Callbacks
    if 'callbacks' in ckpt:
        callbacks = ckpt['callbacks']
        print(f"\n🔔 Callbacks:")
        for callback_name, callback_state in callbacks.items():
            print(f"  • {callback_name}")
    
    # Hyperparameters (if available)
    if 'hyper_parameters' in ckpt:
        print(f"\n⚙️  Hyperparameters:")
        hparams = ckpt['hyper_parameters']
        for key, value in list(hparams.items())[:10]:
            print(f"  • {key}: {value}")
        if len(hparams) > 10:
            print(f"  ... and {len(hparams) - 10} more")
    
    print("\n" + "=" * 70)
    print("✅ Inspection Complete")
    print("=" * 70)
    
    # Resume instructions
    print("\n💡 To resume training from this checkpoint:")
    print("\n1️⃣  Simplest way (PyTorch Lightning):")
    print("   trainer.fit(model, datamodule, ckpt_path='checkpoints/base/base_model_last.ckpt')")
    
    print("\n2️⃣  With checkpoint management:")
    print("   python resume_base_training.py")
    
    print("\n3️⃣  Manual loading:")
    print("   ckpt = torch.load('checkpoints/base/base_model_last.ckpt')")
    print("   model.load_state_dict(ckpt['state_dict'])")
    
    return ckpt


if __name__ == "__main__":
    import sys
    
    # Allow custom checkpoint path
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "checkpoints/base/base_model_last.ckpt"
    
    inspect_checkpoint(checkpoint_path)
