"""
PyTorch Lightning Callbacks for Distributed Checkpointing
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Optional, Dict, Any
from ..utils.checkpoint_manager import CheckpointManager


class DistributedCheckpointCallback(Callback):
    """
    PyTorch Lightning callback for distributed LoRA checkpointing.
    
    Automatically saves:
    - Base model (once at start)
    - LoRA weights (every N epochs)
    - Best checkpoint based on monitored metric
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_every_n_epochs: int = 1,
        save_base_model: bool = True,
        keep_last_n: int = 5,
        verbose: bool = True
    ):
        """
        Initialize distributed checkpoint callback.
        
        Args:
            checkpoint_manager: CheckpointManager instance
            monitor: Metric to monitor for best checkpoint
            mode: 'min' or 'max' for monitored metric
            save_every_n_epochs: Save checkpoint every N epochs
            save_base_model: Save base model at start
            keep_last_n: Number of recent checkpoints to keep
            verbose: Print checkpoint information
        """
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.monitor = monitor
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs
        self.save_base_model = save_base_model
        self.keep_last_n = keep_last_n
        self.verbose = verbose
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.base_model_saved = False
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save base model at the start of training."""
        if self.save_base_model and not self.base_model_saved:
            if self.verbose:
                print("\n💾 Saving base model...")
            
            # Get model state dict
            state_dict = pl_module.state_dict()
            
            # Save base model
            self.checkpoint_manager.save_base_model(
                state_dict,
                config={
                    'model_type': getattr(pl_module, 'model_type', 'unknown'),
                    'img_size': getattr(pl_module, 'img_size', None),
                    'd_model': getattr(pl_module, 'd_model', None),
                }
            )
            
            self.base_model_saved = True
            
            if self.verbose:
                print(self.checkpoint_manager.get_run_summary())
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save LoRA checkpoint at the end of each epoch."""
        current_epoch = trainer.current_epoch
        
        # Check if we should save this epoch
        if (current_epoch + 1) % self.save_every_n_epochs != 0:
            return
        
        # Get metrics
        metrics = {}
        if hasattr(trainer, 'logged_metrics'):
            metrics = {k: float(v) for k, v in trainer.logged_metrics.items()}
        
        # Check if this is the best checkpoint
        is_best = False
        if self.monitor in metrics:
            current_metric = metrics[self.monitor]
            if self._is_better(current_metric, self.best_metric):
                self.best_metric = current_metric
                is_best = True
        
        # Get optimizer and scheduler states
        optimizer_state = None
        scheduler_state = None
        
        if len(trainer.optimizers) > 0:
            optimizer_state = trainer.optimizers[0].state_dict()
        
        if len(trainer.lr_scheduler_configs) > 0:
            scheduler_state = trainer.lr_scheduler_configs[0].scheduler.state_dict()
        
        # Save distributed checkpoint
        if self.verbose:
            print(f"\n💾 Saving checkpoint for epoch {current_epoch + 1}...")
        
        self.checkpoint_manager.save_distributed_checkpoint(
            model=pl_module,
            epoch=current_epoch + 1,
            metrics=metrics,
            is_best=is_best,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            save_base=False  # Base model already saved
        )
        
        # Cleanup old checkpoints
        if self.keep_last_n > 0:
            self.checkpoint_manager.cleanup_old_checkpoints(
                keep_last_n=self.keep_last_n,
                keep_best=True
            )
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Print summary at the end of training."""
        if self.verbose:
            print("\n" + "="*60)
            print("🎉 Training completed!")
            print("="*60)
            print(self.checkpoint_manager.get_run_summary())
            
            best_info = self.checkpoint_manager.get_best_checkpoint_info()
            if best_info:
                print(f"\n⭐ Best checkpoint:")
                print(f"   Epoch: {best_info['epoch']}")
                print(f"   Metrics: {best_info.get('metrics', {})}")
                print(f"   Path: {best_info['path']}")


class ResumeFromCheckpointCallback(Callback):
    """
    Callback to resume training from a checkpoint.
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        verbose: bool = True
    ):
        """
        Initialize resume callback.
        
        Args:
            checkpoint_manager: CheckpointManager instance
            load_optimizer: Load optimizer state
            load_scheduler: Load scheduler state
            verbose: Print resume information
        """
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.load_optimizer = load_optimizer
        self.load_scheduler = load_scheduler
        self.verbose = verbose
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Load checkpoint at the start of training."""
        try:
            if self.verbose:
                print("\n📥 Resuming from checkpoint...")
            
            # Load checkpoint for resume
            resume_dict = self.checkpoint_manager.load_for_resume(
                load_optimizer=self.load_optimizer,
                load_scheduler=self.load_scheduler
            )
            
            # Load LoRA weights into model
            pl_module.load_state_dict(resume_dict['lora_state_dict'], strict=False)
            
            # Load optimizer state
            if self.load_optimizer and 'optimizer_state_dict' in resume_dict:
                if len(trainer.optimizers) > 0:
                    trainer.optimizers[0].load_state_dict(resume_dict['optimizer_state_dict'])
            
            # Load scheduler state
            if self.load_scheduler and 'scheduler_state_dict' in resume_dict:
                if len(trainer.lr_scheduler_configs) > 0:
                    trainer.lr_scheduler_configs[0].scheduler.load_state_dict(
                        resume_dict['scheduler_state_dict']
                    )
            
            # Set starting epoch
            trainer.fit_loop.epoch_loop._batches_that_stepped = resume_dict['epoch']
            
            if self.verbose:
                print(f"✅ Resumed from epoch {resume_dict['epoch']}")
                print(f"   Previous metrics: {resume_dict.get('metrics', {})}")
        
        except FileNotFoundError as e:
            if self.verbose:
                print(f"⚠️  No checkpoint found to resume from: {e}")
                print("   Starting training from scratch...")
