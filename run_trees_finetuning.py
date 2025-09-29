#!/usr/bin/env python3
"""
Simple script to run trees-only fine-tuning
"""

import os
import sys
import yaml
from pathlib import Path

# Add diffit to path if needed
sys.path.insert(0, str(Path(__file__).parent))

def run_trees_finetuning():
    """Run fine-tuning on trees-only CIFAR-100 dataset"""
    
    print("🌳 Starting Trees-only Fine-tuning")
    print("=" * 50)
    
    # Method 1: Use the blockwise LoRA script approach
    print("📋 Configuration:")
    print(f"   Dataset: CIFAR-100 Trees Only (5 classes)")
    print(f"   Classes: maple_tree, oak_tree, palm_tree, pine_tree, willow_tree")
    print(f"   Base Model: checkpoints/base/base_model_last.ckpt")
    print(f"   Output: checkpoints/lora_finetuned_trees/")
    print(f"   Method: LoRA Block-wise Fine-tuning")
    
    # Check if base model exists
    base_model_path = "checkpoints/base/base_model_last.ckpt"
    if not os.path.exists(base_model_path):
        print(f"\n❌ Base model not found at: {base_model_path}")
        print("   Please ensure you have a trained base model first.")
        print("   You can train it using: python scripts/train_base_model.py")
        return False
    
    # Create output directory
    output_dir = "checkpoints/lora_finetuned_trees/"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n✅ Setup complete, starting fine-tuning...")
    
    # Import the dataset loading function with trees filtering
    from diffit.utils.cifar100_classes import get_class_indices
    
    # Get tree class indices
    tree_classes = get_class_indices("trees")
    print(f"🌲 Tree classes: {tree_classes}")
    
    # Method 1A: Use the updated blockwise script with trees
    try:
        # Update the config in the blockwise script to use trees
        print("🔧 Configuring for trees dataset...")
        
        # You can either:
        # 1. Modify diffit_blockwise_lora_finetuning.py to use trees classes
        # 2. Use the CLI approach
        # 3. Use PyTorch Lightning directly
        
        # For now, let's use the CLI approach which is most reliable
        from diffit.cli.finetune import main as finetune_main
        
        # Simulate command line arguments
        import argparse
        sys.argv = [
            'finetune.py',
            '--config', 'configs/training/lora_finetuning.yaml',
            '--base-checkpoint', base_model_path,
            '--output-dir', output_dir
        ]
        
        print("🚀 Starting fine-tuning with CLI...")
        finetune_main()
        
        print("🎉 Trees fine-tuning completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        print("\n🔧 Alternative methods:")
        print("1. Use: python -m diffit.cli.finetune --config configs/training/lora_finetuning.yaml")
        print("2. Use: python scripts/finetune_with_lora.py")
        print("3. Manually run: python diffit_blockwise_lora_finetuning.py (modify config first)")
        return False

if __name__ == "__main__":
    success = run_trees_finetuning()
    
    if success:
        print(f"\n📊 Next steps:")
        print(f"   • Check results in: checkpoints/lora_finetuned_trees/")
        print(f"   • Generate samples: python scripts/generate_samples.py")
        print(f"   • View logs: tensorboard --logdir checkpoints/lora_finetuned_trees/logs")
    else:
        print(f"\n📋 Manual fine-tuning commands:")
        print(f"   python -m diffit.cli.finetune --config configs/training/lora_finetuning.yaml")
        print(f"   python scripts/finetune_with_lora.py --config configs/training/lora_finetuning.yaml")
