"""
DiffiT Training Pipeline

Main training orchestration with exact algorithm preservation.
"""

import pytorch_lightning as pl
from typing import Dict, Any, Optional
import torch

from ..models import UShapedNetwork
# from ..models import LatentDiffiTNetwork  # TODO: Implement when needed
from ..lora import inject_blockwise_lora
from ..utils import load_config, get_device
from ..utils.drive_integration import CheckpointManager
from .data import DiffiTDataModule
from .callbacks import setup_callbacks, setup_logger


class DiffiTTrainer:
    """
    Main trainer class for DiffiT models
    
    Orchestrates training with exact algorithm preservation from original implementation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to training configuration YAML file
        """
        self.config = load_config(config_path)
        self.device = get_device(self.config.get('device'))
        
        # Setup checkpoint manager
        project_name = self.config.get('project_name', 'diffit-lora')
        self.checkpoint_manager = CheckpointManager(project_name=project_name)
        
        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_trainer()
    
    def setup_model(self):
        """Setup the DiffiT model based on configuration"""
        model_config_path = self.config['training']['model_config']
        model_config = load_config(model_config_path)['model']
        
        # Create model based on type
        if model_config['type'] == "image-space":
            self.model = UShapedNetwork(
                learning_rate=model_config['learning_rate'],
                d_model=model_config['d_model'],
                num_heads=model_config['num_heads'],
                dropout=model_config['dropout'],
                d_ff=model_config['d_ff'],
                img_size=model_config['img_size'],
                device=self.device,
                denoising_steps=model_config['denoising_steps'],
                L1=model_config.get('L1', 2),
                L2=model_config.get('L2', 2),
                L3=model_config.get('L3', 2),
                L4=model_config.get('L4', 2)
            )
        elif model_config['type'] == "latent-space":
            # TODO: Implement LatentDiffiTNetwork when needed
            raise NotImplementedError(
                "LatentDiffiTNetwork is not yet implemented. "
                "Currently only UShapedNetwork (image-space) is available. "
                "Please use model type 'image-space' or implement LatentDiffiTNetwork."
            )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        self.model = self.model.to(self.device)
        print(f"✅ Created {model_config['type']} DiffiT model")
    
    def setup_data(self):
        """Setup data module"""
        self.data_module = DiffiTDataModule(self.config)
        print("✅ Data module configured")
    
    def setup_trainer(self):
        """Setup PyTorch Lightning trainer"""
        training_config = self.config['training']
        
        # Setup callbacks and logger
        callbacks = setup_callbacks(self.config)
        logger = setup_logger(self.config)
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=training_config['num_epochs'],
            accelerator=training_config.get('accelerator', 'auto'),
            devices=training_config.get('devices', 'auto'),
            precision=training_config.get('precision', '32-true'),
            gradient_clip_val=training_config.get('gradient_clip_val', 1.0),
            callbacks=callbacks,
            logger=logger,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            check_val_every_n_epoch=training_config.get('check_val_every_n_epoch', 1),
            log_every_n_steps=training_config.get('log_every_n_steps', 25),
        )
        print("✅ PyTorch Lightning trainer configured")
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained model weights"""
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                # PyTorch Lightning checkpoint
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                # Regular PyTorch checkpoint
                self.model.load_state_dict(checkpoint, strict=False)
            
            print(f"✅ Loaded pretrained weights from {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Could not load pretrained weights: {e}")
            print("Continuing with randomly initialized weights")
    
    def inject_lora(self, lora_config_path: str):
        """Inject LoRA adapters for fine-tuning"""
        lora_config = load_config(lora_config_path)['lora']
        
        # Count original parameters
        original_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Before LoRA: {original_params:,} parameters")
        
        # Inject LoRA
        replacements = inject_blockwise_lora(self.model, lora_config)
        
        # Count LoRA parameters
        from ..lora import calculate_lora_parameters
        lora_stats = calculate_lora_parameters(self.model)
        
        print(f"📊 After LoRA: {lora_stats['lora_parameters']:,} trainable parameters")
        print(f"📈 Parameter efficiency: {lora_stats['lora_ratio']:.2f}%")
        
        return lora_stats
    
    def fit(self):
        """Train the model"""
        print("🚀 Starting training...")
        self.trainer.fit(self.model, self.data_module)
        print("✅ Training completed!")
    
    def test(self):
        """Test the model"""
        print("🧪 Starting testing...")
        self.trainer.test(self.model, self.data_module)
        print("✅ Testing completed!")
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        self.trainer.save_checkpoint(path)
        print(f"💾 Model saved to {path}")


class LoRAFineTuner(DiffiTTrainer):
    """
    Specialized trainer for LoRA fine-tuning
    
    Extends DiffiTTrainer with LoRA-specific functionality.
    """
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Load pretrained model if specified
        pretrained_path = self.config['training'].get('pretrained_path')
        if pretrained_path:
            self.load_pretrained(pretrained_path)
        
        # Inject LoRA adapters
        lora_config_path = self.config['training']['lora_config']
        self.lora_stats = self.inject_lora(lora_config_path)
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights"""
        from ..lora import save_lora_weights
        save_lora_weights(self.model, path)
        print(f"💾 LoRA weights saved to {path}")
    
    def fuse_lora(self):
        """Fuse LoRA weights into base model"""
        from ..lora import fuse_all_lora
        fuse_all_lora(self.model)
        print("🔗 LoRA weights fused into base model")
    
    def save_checkpoint_to_drive(self, filename: str, epoch: int = None, metrics: dict = None):
        """
        Save checkpoint with Drive backup
        
        Args:
            filename: Name for the checkpoint file
            epoch: Current epoch (optional)
            metrics: Training metrics (optional)
        """
        # Prepare checkpoint data
        checkpoint_data = {
            'state_dict': self.model.state_dict(),
            'model_config': self.config,
            'epoch': epoch,
            'metrics': metrics or {},
        }
        
        # Add LoRA-specific information if applicable
        if hasattr(self, 'lora_stats') and self.lora_stats:
            checkpoint_data['lora_stats'] = self.lora_stats
            checkpoint_data['is_lora_model'] = True
        
        # Save with checkpoint manager
        saved_path = self.checkpoint_manager.save_checkpoint(checkpoint_data, filename)
        print(f"💾 Checkpoint saved: {filename}")
        return saved_path
    
    def load_checkpoint_from_drive(self, filename: str, strict: bool = True):
        """
        Load checkpoint with Drive/local fallback
        
        Args:
            filename: Name of the checkpoint file
            strict: Whether to enforce strict state dict loading
            
        Returns:
            Loaded checkpoint data or None
        """
        checkpoint_data = self.checkpoint_manager.load_checkpoint(filename)
        
        if checkpoint_data is not None:
            try:
                self.model.load_state_dict(checkpoint_data['state_dict'], strict=strict)
                print(f"✅ Model state loaded from: {filename}")
                return checkpoint_data
            except Exception as e:
                print(f"❌ Failed to load model state: {e}")
                return None
        else:
            print(f"❌ Checkpoint not found: {filename}")
            return None
    
    def list_available_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        print("📋 Available Checkpoints:")
        if checkpoints['local']:
            print("  Local:")
            for ckpt in checkpoints['local']:
                print(f"    - {ckpt}")
        
        if checkpoints['drive']:
            print("  Google Drive:")
            for ckpt in checkpoints['drive']:
                print(f"    - {ckpt}")
        
        if not checkpoints['local'] and not checkpoints['drive']:
            print("  No checkpoints found")
        
        return checkpoints
    
    def sync_checkpoints_with_drive(self, direction: str = "both"):
        """
        Sync checkpoints between Drive and local storage
        
        Args:
            direction: "drive_to_local", "local_to_drive", or "both"
        """
        self.checkpoint_manager.sync_checkpoints(direction)
    
    def download_checkpoint(self, filename: str):
        """Download checkpoint to local machine (Colab only)"""
        self.checkpoint_manager.download_checkpoint(filename)
