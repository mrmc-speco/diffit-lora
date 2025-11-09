"""
Checkpoint Manager for DiffiT with LoRA

Manages distributed checkpointing with separate LoRA and base model weights,
organized by run numbers for easy resumption of training.
"""

import os
import json
import torch
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import glob


class CheckpointManager:
    """
    Manages distributed checkpoints with run numbering system.
    
    Directory structure:
    checkpoints/
    ├── run_001/
    │   ├── config.json
    │   ├── base_model/
    │   │   └── base_model.ckpt
    │   ├── lora_weights/
    │   │   ├── epoch_01.ckpt
    │   │   ├── epoch_02.ckpt
    │   │   └── best.ckpt
    │   └── metadata.json
    ├── run_002/
    │   └── ...
    """
    
    def __init__(
        self,
        base_dir: str = "checkpoints",
        experiment_name: str = "diffit_lora",
        run_number: Optional[int] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            base_dir: Base directory for all checkpoints
            experiment_name: Name of the experiment
            run_number: Specific run number (auto-increments if None)
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine run number
        if run_number is None:
            self.run_number = self._get_next_run_number()
        else:
            self.run_number = run_number
        
        # Set up run directory
        self.run_dir = self.base_dir / f"run_{self.run_number:03d}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.base_model_dir = self.run_dir / "base_model"
        self.lora_weights_dir = self.run_dir / "lora_weights"
        self.logs_dir = self.run_dir / "logs"
        
        self.base_model_dir.mkdir(exist_ok=True)
        self.lora_weights_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Metadata tracking
        self.metadata_file = self.run_dir / "metadata.json"
        self.metadata = self._load_or_create_metadata()
    
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
                'training_config': {},
                'lora_config': {}
            }
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_base_model(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        config: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ):
        """
        Save base model weights (without LoRA).
        
        Args:
            model_state_dict: State dict of the base model
            config: Model configuration
            overwrite: Whether to overwrite existing base model
        """
        base_model_path = self.base_model_dir / "base_model.ckpt"
        
        if base_model_path.exists() and not overwrite:
            print(f"⚠️  Base model already exists at {base_model_path}")
            print("   Use overwrite=True to replace it")
            return str(base_model_path)
        
        # Filter out LoRA parameters from state dict
        base_state_dict = {
            k: v for k, v in model_state_dict.items()
            if not ('lora_A' in k or 'lora_B' in k or 'lora_' in k)
        }
        
        checkpoint = {
            'state_dict': base_state_dict,
            'config': config or {},
            'saved_at': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, base_model_path)
        print(f"✅ Base model saved to: {base_model_path}")
        print(f"   Parameters: {len(base_state_dict):,} tensors")
        
        return str(base_model_path)
    
    def save_lora_weights(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save LoRA weights separately.
        
        Args:
            model_state_dict: Full model state dict
            epoch: Current epoch number
            metrics: Training metrics
            is_best: Whether this is the best checkpoint
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            
        Returns:
            Path to saved checkpoint
        """
        # Extract only LoRA parameters
        lora_state_dict = {
            k: v for k, v in model_state_dict.items()
            if 'lora_A' in k or 'lora_B' in k or 'lora_' in k
        }
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'lora_state_dict': lora_state_dict,
            'metrics': metrics or {},
            'saved_at': datetime.now().isoformat()
        }
        
        # Add optimizer and scheduler states if provided
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        # Save epoch checkpoint
        epoch_path = self.lora_weights_dir / f"epoch_{epoch:03d}.ckpt"
        torch.save(checkpoint, epoch_path)
        
        # Update metadata
        self.metadata['checkpoints'].append({
            'epoch': epoch,
            'path': str(epoch_path),
            'metrics': metrics or {},
            'saved_at': checkpoint['saved_at']
        })
        
        print(f"✅ LoRA weights saved to: {epoch_path}")
        print(f"   Epoch: {epoch}, LoRA parameters: {len(lora_state_dict):,} tensors")
        
        # Save as best if specified
        if is_best:
            best_path = self.lora_weights_dir / "best.ckpt"
            shutil.copy2(epoch_path, best_path)
            self.metadata['best_checkpoint'] = {
                'epoch': epoch,
                'path': str(best_path),
                'metrics': metrics or {}
            }
            print(f"⭐ Best checkpoint updated: {best_path}")
        
        # Save last checkpoint
        last_path = self.lora_weights_dir / "last.ckpt"
        shutil.copy2(epoch_path, last_path)
        
        self._save_metadata()
        
        return str(epoch_path)
    
    def save_distributed_checkpoint(
        self,
        model,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        save_base: bool = False
    ):
        """
        Save distributed checkpoint (LoRA + optionally base model).
        
        Args:
            model: The model to save
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best checkpoint
            optimizer_state: Optimizer state
            scheduler_state: Scheduler state
            save_base: Whether to save base model too
        """
        model_state_dict = model.state_dict()
        
        # Save LoRA weights
        lora_path = self.save_lora_weights(
            model_state_dict,
            epoch,
            metrics,
            is_best,
            optimizer_state,
            scheduler_state
        )
        
        # Optionally save base model
        if save_base:
            self.save_base_model(model_state_dict)
        
        return lora_path
    
    def load_base_model(self) -> Dict[str, Any]:
        """Load base model checkpoint."""
        base_model_path = self.base_model_dir / "base_model.ckpt"
        
        if not base_model_path.exists():
            raise FileNotFoundError(f"Base model not found at {base_model_path}")
        
        checkpoint = torch.load(base_model_path, map_location='cpu')
        print(f"✅ Loaded base model from: {base_model_path}")
        
        return checkpoint
    
    def load_lora_weights(
        self,
        epoch: Optional[int] = None,
        load_best: bool = False,
        load_last: bool = False
    ) -> Dict[str, Any]:
        """
        Load LoRA weights.
        
        Args:
            epoch: Specific epoch to load
            load_best: Load best checkpoint
            load_last: Load last checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        if load_best:
            ckpt_path = self.lora_weights_dir / "best.ckpt"
        elif load_last:
            ckpt_path = self.lora_weights_dir / "last.ckpt"
        elif epoch is not None:
            ckpt_path = self.lora_weights_dir / f"epoch_{epoch:03d}.ckpt"
        else:
            raise ValueError("Must specify epoch, load_best=True, or load_last=True")
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print(f"✅ Loaded LoRA weights from: {ckpt_path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        
        return checkpoint
    
    def load_for_resume(
        self,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint for resuming training.
        
        Returns:
            Dictionary with all necessary components
        """
        # Load last LoRA checkpoint
        lora_ckpt = self.load_lora_weights(load_last=True)
        
        resume_dict = {
            'lora_state_dict': lora_ckpt['lora_state_dict'],
            'epoch': lora_ckpt['epoch'],
            'metrics': lora_ckpt.get('metrics', {})
        }
        
        if load_optimizer and 'optimizer_state_dict' in lora_ckpt:
            resume_dict['optimizer_state_dict'] = lora_ckpt['optimizer_state_dict']
        
        if load_scheduler and 'scheduler_state_dict' in lora_ckpt:
            resume_dict['scheduler_state_dict'] = lora_ckpt['scheduler_state_dict']
        
        print(f"📥 Prepared for resuming from epoch {resume_dict['epoch']}")
        
        return resume_dict
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints for this run."""
        return self.metadata.get('checkpoints', [])
    
    def get_best_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the best checkpoint."""
        return self.metadata.get('best_checkpoint')
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5, keep_best: bool = True):
        """
        Clean up old checkpoints, keeping only the last N.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
            keep_best: Whether to always keep the best checkpoint
        """
        checkpoints = sorted(
            self.metadata.get('checkpoints', []),
            key=lambda x: x['epoch']
        )
        
        if len(checkpoints) <= keep_last_n:
            return
        
        best_epoch = None
        if keep_best and self.metadata.get('best_checkpoint'):
            best_epoch = self.metadata['best_checkpoint']['epoch']
        
        to_remove = checkpoints[:-keep_last_n]
        removed_count = 0
        
        for ckpt in to_remove:
            if keep_best and ckpt['epoch'] == best_epoch:
                continue
            
            ckpt_path = Path(ckpt['path'])
            if ckpt_path.exists():
                ckpt_path.unlink()
                removed_count += 1
        
        # Update metadata
        self.metadata['checkpoints'] = [
            c for c in checkpoints
            if c['epoch'] >= checkpoints[-keep_last_n]['epoch'] or
            (keep_best and c['epoch'] == best_epoch)
        ]
        self._save_metadata()
        
        print(f"🗑️  Cleaned up {removed_count} old checkpoints")
    
    def save_config(self, training_config: Dict[str, Any], lora_config: Dict[str, Any]):
        """Save training and LoRA configurations."""
        self.metadata['training_config'] = training_config
        self.metadata['lora_config'] = lora_config
        
        # Also save as separate files for easy reference
        config_file = self.run_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'training': training_config,
                'lora': lora_config
            }, f, indent=2)
        
        self._save_metadata()
        print(f"✅ Configurations saved to: {config_file}")
    
    def get_run_summary(self) -> str:
        """Get a summary of the current run."""
        best_ckpt = self.metadata.get('best_checkpoint')
        best_epoch = best_ckpt.get('epoch', 'N/A') if best_ckpt else 'N/A'
        
        summary = f"""
╔══════════════════════════════════════════════════════════╗
║  Checkpoint Manager - Run {self.run_number:03d}
╠══════════════════════════════════════════════════════════╣
║  Experiment: {self.experiment_name}
║  Run Directory: {self.run_dir}
║  Created: {self.metadata.get('created_at', 'unknown')}
║  
║  Checkpoints: {len(self.metadata.get('checkpoints', []))}
║  Best Checkpoint: Epoch {best_epoch}
║  
║  Directories:
║    • Base Model: {self.base_model_dir}
║    • LoRA Weights: {self.lora_weights_dir}
║    • Logs: {self.logs_dir}
╚══════════════════════════════════════════════════════════╝
"""
        return summary


def get_run_manager(
    base_dir: str = "checkpoints",
    run_number: Optional[int] = None,
    experiment_name: str = "diffit_lora"
) -> CheckpointManager:
    """
    Factory function to get checkpoint manager.
    
    Args:
        base_dir: Base checkpoint directory
        run_number: Specific run number (None for new run)
        experiment_name: Experiment name
        
    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(base_dir, experiment_name, run_number)


def list_all_runs(base_dir: str = "checkpoints") -> List[Dict[str, Any]]:
    """
    List all available runs.
    
    Args:
        base_dir: Base checkpoint directory
        
    Returns:
        List of run information
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
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
                    'path': str(run_dir)
                })
    
    return runs
