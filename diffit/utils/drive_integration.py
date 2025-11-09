"""
Google Drive Integration for Colab
Automatic checkpoint loading/saving with Google Drive support and local fallback.
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Union
import torch

try:
    from google.colab import drive, files
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

from .logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages checkpoint loading/saving with Google Drive integration for Colab
    """
    
    def __init__(self, 
                 project_name: str = "diffit-lora",
                 drive_path: str = "/content/drive/MyDrive",
                 local_checkpoint_dir: str = "./checkpoints"):
        """
        Initialize checkpoint manager
        
        Args:
            project_name: Name of your project folder in Google Drive
            drive_path: Path to Google Drive mount point in Colab
            local_checkpoint_dir: Local checkpoint directory fallback
        """
        self.project_name = project_name
        self.drive_path = drive_path
        self.local_checkpoint_dir = Path(local_checkpoint_dir)
        self.is_colab = self._detect_colab()
        self.drive_mounted = False
        
        # Setup paths
        if self.is_colab:
            self.drive_project_path = Path(drive_path) / project_name
            self.drive_checkpoint_path = self.drive_project_path / "checkpoints"
        else:
            self.drive_project_path = None
            self.drive_checkpoint_path = None
        
        # Ensure local directory exists
        self.local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔧 CheckpointManager initialized")
        logger.info(f"   Environment: {'Google Colab' if self.is_colab else 'Local'}")
        logger.info(f"   Local path: {self.local_checkpoint_dir}")
        if self.is_colab:
            logger.info(f"   Drive path: {self.drive_checkpoint_path}")
    
    def _detect_colab(self) -> bool:
        """Detect if running in Google Colab"""
        return COLAB_AVAILABLE and 'COLAB_GPU' in os.environ
    
    def mount_drive(self) -> bool:
        """
        Mount Google Drive (Colab only)
        
        Returns:
            True if successful or already mounted, False otherwise
        """
        if not self.is_colab:
            logger.warning("⚠️ Drive mounting only available in Google Colab")
            return False
        
        try:
            if not os.path.exists(self.drive_path):
                logger.info("📁 Mounting Google Drive...")
                drive.mount('/content/drive')
            
            # Create project directory if it doesn't exist
            self.drive_project_path.mkdir(parents=True, exist_ok=True)
            self.drive_checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            self.drive_mounted = True
            logger.info("✅ Google Drive mounted successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to mount Google Drive: {e}")
            return False
    
    def save_checkpoint(self, 
                       checkpoint_data: dict, 
                       filename: str,
                       backup_to_drive: bool = True) -> str:
        """
        Save checkpoint with automatic Drive backup
        
        Args:
            checkpoint_data: Checkpoint dictionary to save
            filename: Filename for the checkpoint
            backup_to_drive: Whether to backup to Google Drive (if available)
            
        Returns:
            Path where checkpoint was saved
        """
        # Ensure filename has .ckpt extension
        if not filename.endswith('.ckpt'):
            filename += '.ckpt'
        
        # Save locally first
        local_path = self.local_checkpoint_dir / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint_data, local_path)
        logger.info(f"💾 Checkpoint saved locally: {local_path}")
        
        # Backup to Drive if requested and available
        if backup_to_drive and self.is_colab:
            if not self.drive_mounted:
                self.mount_drive()
            
            if self.drive_mounted:
                try:
                    drive_path = self.drive_checkpoint_path / filename
                    drive_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local_path, drive_path)
                    logger.info(f"☁️ Checkpoint backed up to Drive: {drive_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to backup to Drive: {e}")
        
        return str(local_path)
    
    def load_checkpoint(self, 
                       filename: str,
                       try_drive_first: bool = True) -> Optional[dict]:
        """
        Load checkpoint with automatic Drive/local fallback
        
        Args:
            filename: Filename of the checkpoint
            try_drive_first: Whether to try Drive first (if available)
            
        Returns:
            Loaded checkpoint dictionary or None if not found
        """
        # Ensure filename has .ckpt extension
        if not filename.endswith('.ckpt'):
            filename += '.ckpt'
        
        checkpoint = None
        
        # Try Drive first if requested and available
        if try_drive_first and self.is_colab:
            if not self.drive_mounted:
                self.mount_drive()
            
            if self.drive_mounted:
                drive_path = self.drive_checkpoint_path / filename
                if drive_path.exists():
                    try:
                        checkpoint = torch.load(drive_path, map_location='cpu')
                        logger.info(f"📥 Checkpoint loaded from Drive: {drive_path}")
                        
                        # Also copy to local for faster future access
                        local_path = self.local_checkpoint_dir / filename
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(drive_path, local_path)
                        logger.info(f"📋 Checkpoint cached locally: {local_path}")
                        
                        return checkpoint
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load from Drive: {e}")
        
        # Fallback to local
        local_path = self.local_checkpoint_dir / filename
        if local_path.exists():
            try:
                checkpoint = torch.load(local_path, map_location='cpu')
                logger.info(f"📥 Checkpoint loaded locally: {local_path}")
                return checkpoint
            except Exception as e:
                logger.error(f"❌ Failed to load local checkpoint: {e}")
        
        logger.error(f"❌ Checkpoint not found: {filename}")
        return None
    
    def list_checkpoints(self, location: str = "both") -> dict:
        """
        List available checkpoints
        
        Args:
            location: "local", "drive", or "both"
            
        Returns:
            Dictionary with checkpoint lists
        """
        result = {"local": [], "drive": []}
        
        # List local checkpoints
        if location in ["local", "both"]:
            if self.local_checkpoint_dir.exists():
                result["local"] = [
                    f.name for f in self.local_checkpoint_dir.rglob("*.ckpt")
                ]
        
        # List drive checkpoints
        if location in ["drive", "both"] and self.is_colab:
            if not self.drive_mounted:
                self.mount_drive()
            
            if self.drive_mounted and self.drive_checkpoint_path.exists():
                result["drive"] = [
                    f.name for f in self.drive_checkpoint_path.rglob("*.ckpt")
                ]
        
        return result
    
    def sync_checkpoints(self, direction: str = "drive_to_local"):
        """
        Sync checkpoints between Drive and local storage
        
        Args:
            direction: "drive_to_local", "local_to_drive", or "both"
        """
        if not self.is_colab:
            logger.warning("⚠️ Sync only available in Google Colab")
            return
        
        if not self.drive_mounted:
            self.mount_drive()
        
        if not self.drive_mounted:
            logger.error("❌ Cannot sync: Drive not mounted")
            return
        
        try:
            if direction in ["drive_to_local", "both"]:
                # Sync from Drive to local
                if self.drive_checkpoint_path.exists():
                    for drive_file in self.drive_checkpoint_path.rglob("*.ckpt"):
                        local_file = self.local_checkpoint_dir / drive_file.name
                        if not local_file.exists() or drive_file.stat().st_mtime > local_file.stat().st_mtime:
                            local_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(drive_file, local_file)
                            logger.info(f"📥 Synced from Drive: {drive_file.name}")
            
            if direction in ["local_to_drive", "both"]:
                # Sync from local to Drive
                if self.local_checkpoint_dir.exists():
                    for local_file in self.local_checkpoint_dir.rglob("*.ckpt"):
                        drive_file = self.drive_checkpoint_path / local_file.name
                        if not drive_file.exists() or local_file.stat().st_mtime > drive_file.stat().st_mtime:
                            drive_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(local_file, drive_file)
                            logger.info(f"☁️ Synced to Drive: {local_file.name}")
            
            logger.info("✅ Checkpoint sync completed")
            
        except Exception as e:
            logger.error(f"❌ Sync failed: {e}")
    
    def download_checkpoint(self, filename: str):
        """
        Download checkpoint to local machine (Colab only)
        
        Args:
            filename: Name of checkpoint file to download
        """
        if not self.is_colab:
            logger.warning("⚠️ Download only available in Google Colab")
            return
        
        local_path = self.local_checkpoint_dir / filename
        if local_path.exists():
            try:
                files.download(str(local_path))
                logger.info(f"⬇️ Downloaded: {filename}")
            except Exception as e:
                logger.error(f"❌ Download failed: {e}")
        else:
            logger.error(f"❌ File not found: {filename}")
    
    def create_checkpoint_archive(self, archive_name: str = None) -> str:
        """
        Create a zip archive of all checkpoints
        
        Args:
            archive_name: Name for the archive (optional)
            
        Returns:
            Path to created archive
        """
        if archive_name is None:
            archive_name = f"{self.project_name}_checkpoints.zip"
        
        archive_path = self.local_checkpoint_dir.parent / archive_name
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for checkpoint_file in self.local_checkpoint_dir.rglob("*.ckpt"):
                zipf.write(checkpoint_file, checkpoint_file.relative_to(self.local_checkpoint_dir))
        
        logger.info(f"📦 Checkpoint archive created: {archive_path}")
        return str(archive_path)


# Convenience functions for easy usage
def get_checkpoint_manager(project_name: str = "diffit-lora") -> CheckpointManager:
    """Get a configured checkpoint manager"""
    return CheckpointManager(project_name=project_name)


def save_checkpoint(checkpoint_data: dict, filename: str, project_name: str = "diffit-lora") -> str:
    """Quick checkpoint save with Drive backup"""
    manager = get_checkpoint_manager(project_name)
    return manager.save_checkpoint(checkpoint_data, filename)


def load_checkpoint(filename: str, project_name: str = "diffit-lora") -> Optional[dict]:
    """Quick checkpoint load with Drive/local fallback"""
    manager = get_checkpoint_manager(project_name)
    return manager.load_checkpoint(filename)
