#!/usr/bin/env python3
"""
Simple example demonstrating Google Drive checkpoint management
"""

import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffit.utils.drive_integration import CheckpointManager, save_checkpoint, load_checkpoint
from diffit.models import UShapedNetwork


def main():
    """Demonstrate checkpoint management with Google Drive integration"""
    
    print("🚀 DiffiT Google Drive Checkpoint Management Demo")
    print("=" * 50)
    
    # Initialize checkpoint manager
    print("1. Initializing checkpoint manager...")
    checkpoint_manager = CheckpointManager(project_name="diffit-lora-demo")
    
    # Check environment
    if checkpoint_manager.is_colab:
        print("   ✅ Running in Google Colab")
        # Mount drive
        if checkpoint_manager.mount_drive():
            print("   ✅ Google Drive mounted successfully")
        else:
            print("   ⚠️ Drive mounting failed, using local storage only")
    else:
        print("   💻 Running locally")
    
    print(f"   📂 Local path: {checkpoint_manager.local_checkpoint_dir}")
    if checkpoint_manager.drive_checkpoint_path:
        print(f"   ☁️ Drive path: {checkpoint_manager.drive_checkpoint_path}")
    
    # Create a simple model for demonstration
    print("\n2. Creating demo model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🔥 Using device: {device}")
    
    model = UShapedNetwork(
        learning_rate=0.001,
        d_model=64,
        num_heads=2,
        dropout=0.1,
        d_ff=128,
        img_size=32,
        device=device,
        denoising_steps=100,
        L1=1, L2=1, L3=1, L4=1
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   🤖 Created model with {param_count:,} parameters")
    
    # Save checkpoint
    print("\n3. Saving checkpoint...")
    checkpoint_data = {
        'state_dict': model.state_dict(),
        'epoch': 1,
        'metrics': {'loss': 0.8, 'lr': 0.001},
        'model_config': {
            'type': 'image-space',
            'd_model': 64,
            'num_heads': 2,
            'img_size': 32
        }
    }
    
    saved_path = checkpoint_manager.save_checkpoint(
        checkpoint_data, 
        "demo_model_v1",
        backup_to_drive=True
    )
    print(f"   💾 Checkpoint saved: {saved_path}")
    
    # List checkpoints
    print("\n4. Listing available checkpoints...")
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if checkpoints['local']:
        print("   📂 Local checkpoints:")
        for ckpt in checkpoints['local']:
            print(f"     - {ckpt}")
    
    if checkpoints['drive']:
        print("   ☁️ Drive checkpoints:")
        for ckpt in checkpoints['drive']:
            print(f"     - {ckpt}")
    
    if not checkpoints['local'] and not checkpoints['drive']:
        print("   ❌ No checkpoints found")
    
    # Load checkpoint
    print("\n5. Loading checkpoint...")
    loaded_checkpoint = checkpoint_manager.load_checkpoint("demo_model_v1")
    
    if loaded_checkpoint:
        print("   ✅ Checkpoint loaded successfully!")
        print(f"     Epoch: {loaded_checkpoint.get('epoch')}")
        print(f"     Metrics: {loaded_checkpoint.get('metrics')}")
        
        # Load state into model
        model.load_state_dict(loaded_checkpoint['state_dict'])
        print("   🤖 Model state restored")
    else:
        print("   ❌ Failed to load checkpoint")
    
    # Sync checkpoints (Colab only)
    if checkpoint_manager.is_colab and checkpoint_manager.drive_mounted:
        print("\n6. Syncing checkpoints with Drive...")
        checkpoint_manager.sync_checkpoints(direction="both")
        print("   🔄 Sync completed")
    
    # Quick save/load example
    print("\n7. Quick save/load example...")
    quick_data = {
        'state_dict': model.state_dict(),
        'note': 'Quick save example'
    }
    
    # Using convenience functions
    save_checkpoint(quick_data, "quick_demo", project_name="diffit-lora-demo")
    loaded_quick = load_checkpoint("quick_demo", project_name="diffit-lora-demo")
    
    if loaded_quick:
        print(f"   ✅ Quick save/load: {loaded_quick.get('note')}")
    
    # Create archive
    print("\n8. Creating checkpoint archive...")
    archive_path = checkpoint_manager.create_checkpoint_archive("demo_backup.zip")
    print(f"   📦 Archive created: {archive_path}")
    
    # Download example (Colab only)
    if checkpoint_manager.is_colab:
        print("\n9. Download example...")
        print("   ⬇️ You can download checkpoints using:")
        print("   checkpoint_manager.download_checkpoint('demo_model_v1.ckpt')")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("- Use checkpoint_manager.save_checkpoint() in your training loops")
    print("- Load checkpoints with checkpoint_manager.load_checkpoint()")
    print("- Sync regularly with checkpoint_manager.sync_checkpoints()")
    print("- Create backups with checkpoint_manager.create_checkpoint_archive()")


if __name__ == "__main__":
    main()
