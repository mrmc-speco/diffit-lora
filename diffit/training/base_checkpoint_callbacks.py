"""
PyTorch Lightning Callbacks for Base Model Checkpointing

Unlike LoRA training (which separates base and LoRA weights),
base model training saves the full model at each checkpoint.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Optional, Dict, Any
from pathlib import Path
import torch
import json
from datetime import datetime


class BaseModelCheckpointCallback(Callback):
    """
    PyTorch Lightning callback for base model checkpointing with run management.
    
    Similar to DistributedCheckpointCallback but for full model training.
    Saves complete model state at each epoch.
    """
    
    def __init__(
        self,
        base_dir: str = "checkpoints/base_training",
        run_number: Optional[int] = None,
        experiment_name: str = "diffit_base",
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_every_n_epochs: int = 1,
        keep_last_n: int = 3,
        verbose: bool = True
    ):
        """
        Initialize base model checkpoint callback.
        
        Args:
            base_dir: Base directory for checkpoints
            run_number: Specific run number (auto-increment if None)
            experiment_name: Name of experiment
            monitor: Metric to monitor for best checkpoint
            mode: 'min' or 'max' for monitored metric
            save_every_n_epochs: Save checkpoint every N epochs
            keep_last_n: Number of recent checkpoints to keep
            verbose: Print checkpoint information
        """
        super().__init__()
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.monitor = monitor
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_last_n = keep_last_n
        self.verbose = verbose
        
        # Setup run directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        if run_number is None:
            self.run_number = self._get_next_run_number()
        else:
            self.run_number = run_number
        
        self.run_dir = self.base_dir / f"run_{self.run_number:03d}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint directory
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Logs directory
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Metadata
        self.metadata_file = self.run_dir / "metadata.json"
        self.metadata = self._load_or_create_metadata()
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # Track starting epoch for resume scenarios
        self.starting_epoch = 0
    
    def set_starting_epoch(self, epoch: int):
        """Set the starting epoch for resume scenarios."""
        self.starting_epoch = epoch
        if self.verbose:
            print(f"📊 Starting epoch set to: {epoch}")
    
    def _get_next_run_number(self) -> int:
        """Get the next available run number."""
        existing_runs = list(self.base_dir.glob("run_*"))
        if not existing_runs:
            return 1
        
        run_numbers = []
        for run_dir in existing_runs:
            try:
                num = int(run_dir.name.split("_")[1])
                run_numbers.append(num)
            except (ValueError, IndexError):
                continue
        
        return max(run_numbers) + 1 if run_numbers else 1
    
    def _load_or_create_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create new."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'run_number': self.run_number,
                'experiment_name': self.experiment_name,
                'created_at': datetime.now().isoformat(),
                'checkpoints': [],
                'best_checkpoint': None,
                'training_config': {}
            }
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Print summary at the start of training."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Base Model Training - Run {self.run_number:03d}")
            print(f"{'='*60}")
            print(f"Experiment: {self.experiment_name}")
            print(f"Run Directory: {self.run_dir}")
            print(f"Checkpoint Directory: {self.checkpoint_dir}")
            print(f"Monitoring: {self.monitor} ({self.mode})")
            print(f"{'='*60}\n")
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save checkpoint at the end of each epoch."""
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
        
        # Calculate total epoch (including previously trained epochs)
        total_epoch = current_epoch + 1 + self.starting_epoch
        
        # Create checkpoint
        checkpoint = {
            'epoch': total_epoch,
            'state_dict': pl_module.state_dict(),
            'metrics': metrics,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'saved_at': datetime.now().isoformat()
        }
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"epoch_{total_epoch:03d}.ckpt"
        torch.save(checkpoint, epoch_path)
        
        if self.verbose:
            print(f"\n💾 Saved checkpoint: {epoch_path.name}")
            if metrics:
                print(f"   Metrics: {metrics}")
        
        # Update metadata
        self.metadata['checkpoints'].append({
            'epoch': current_epoch + 1,
            'path': str(epoch_path),
            'metrics': metrics,
            'saved_at': checkpoint['saved_at']
        })
        
        # Save as best if specified
        if is_best:
            best_path = self.checkpoint_dir / "best.ckpt"
            torch.save(checkpoint, best_path)
            self.metadata['best_checkpoint'] = {
                'epoch': total_epoch,
                'path': str(best_path),
                'metrics': metrics
            }
            if self.verbose:
                print(f"   ⭐ New best checkpoint! {self.monitor}={current_metric:.4f}")
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / "last.ckpt"
        torch.save(checkpoint, last_path)
        
        self._save_metadata()
        
        # Cleanup old checkpoints
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.metadata.get('checkpoints', []),
            key=lambda x: x['epoch']
        )
        
        if len(checkpoints) <= self.keep_last_n:
            return
        
        best_epoch = None
        if self.metadata.get('best_checkpoint'):
            best_epoch = self.metadata['best_checkpoint']['epoch']
        
        to_remove = checkpoints[:-self.keep_last_n]
        removed_count = 0
        
        for ckpt in to_remove:
            # Don't remove best checkpoint
            if ckpt['epoch'] == best_epoch:
                continue
            
            ckpt_path = Path(ckpt['path'])
            if ckpt_path.exists():
                ckpt_path.unlink()
                removed_count += 1
        
        # Update metadata
        self.metadata['checkpoints'] = [
            c for c in checkpoints
            if c['epoch'] >= checkpoints[-self.keep_last_n]['epoch'] or
            c['epoch'] == best_epoch
        ]
        self._save_metadata()
        
        if self.verbose and removed_count > 0:
            print(f"   🗑️  Cleaned up {removed_count} old checkpoints")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Print summary at the end of training."""
        if self.verbose:
            print(f"\n{'='*60}")
            print("🎉 Training completed!")
            print(f"{'='*60}")
            print(f"Total checkpoints: {len(self.metadata.get('checkpoints', []))}")
            
            best_info = self.metadata.get('best_checkpoint')
            if best_info:
                print(f"\n⭐ Best checkpoint:")
                print(f"   Epoch: {best_info['epoch']}")
                print(f"   Metrics: {best_info.get('metrics', {})}")
                print(f"   Path: {best_info['path']}")
            
            print(f"\n📁 Run directory: {self.run_dir}")
            print(f"{'='*60}\n")


class ResumeBaseModelCallback(Callback):
    """
    Callback to resume base model training from a checkpoint.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        verbose: bool = True
    ):
        """
        Initialize resume callback.
        
        Args:
            checkpoint_path: Path to checkpoint file (e.g., 'checkpoints/base_training/run_001/checkpoints/last.ckpt')
            load_optimizer: Load optimizer state
            load_scheduler: Load scheduler state
            verbose: Print resume information
        """
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.load_optimizer = load_optimizer
        self.load_scheduler = load_scheduler
        self.verbose = verbose
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Load checkpoint at the start of training."""
        if not self.checkpoint_path.exists():
            if self.verbose:
                print(f"⚠️  Checkpoint not found: {self.checkpoint_path}")
                print("   Starting training from scratch...")
            return
        
        try:
            if self.verbose:
                print(f"\n📥 Resuming from checkpoint: {self.checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Load model state
            pl_module.load_state_dict(checkpoint['state_dict'])
            
            # Load optimizer state
            if self.load_optimizer and 'optimizer_state_dict' in checkpoint:
                if len(trainer.optimizers) > 0:
                    trainer.optimizers[0].load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if self.load_scheduler and 'scheduler_state_dict' in checkpoint:
                if len(trainer.lr_scheduler_configs) > 0:
                    trainer.lr_scheduler_configs[0].scheduler.load_state_dict(
                        checkpoint['scheduler_state_dict']
                    )
            
            # Set starting epoch - this is handled automatically by PyTorch Lightning
            # when using trainer.fit() with ckpt_path parameter
            
            if self.verbose:
                print(f"✅ Resumed from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'metrics' in checkpoint:
                    print(f"   Previous metrics: {checkpoint['metrics']}")
                print()
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to load checkpoint: {e}")
                print("   Starting training from scratch...")


def get_latest_checkpoint(base_dir: str, run_number: int) -> Optional[str]:
    """
    Get the path to the latest checkpoint for a given run.
    
    Args:
        base_dir: Base checkpoint directory
        run_number: Run number
        
    Returns:
        Path to last.ckpt or None if not found
    """
    run_dir = Path(base_dir) / f"run_{run_number:03d}"
    last_ckpt = run_dir / "checkpoints" / "last.ckpt"
    
    if last_ckpt.exists():
        return str(last_ckpt)
    
    return None


def get_best_checkpoint(base_dir: str, run_number: int) -> Optional[str]:
    """
    Get the path to the best checkpoint for a given run.
    
    Args:
        base_dir: Base checkpoint directory
        run_number: Run number
        
    Returns:
        Path to best.ckpt or None if not found
    """
    run_dir = Path(base_dir) / f"run_{run_number:03d}"
    best_ckpt = run_dir / "checkpoints" / "best.ckpt"
    
    if best_ckpt.exists():
        return str(best_ckpt)
    
    return None
