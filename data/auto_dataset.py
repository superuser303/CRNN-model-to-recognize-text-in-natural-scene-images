import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict
from .api_downloader import APIDatasetDownloader  # Relative import

class AutoDataset:
    """Automatically manages dataset availability including download, caching and validation"""
    
    def __init__(self, config_path: str = None):
        self.downloader = APIDatasetDownloader(config_path)
        self.project_root = self.downloader.project_root
        self.cache_dir = self.project_root / "data" / "cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.dataset_checksums = self._load_checksums()

    def get_dataset(self, name: str, force_redownload: bool = False) -> Path:
        """
        Get dataset path, automatically downloading if needed
        :param name: Dataset name (e.g., "IIIT5K")
        :param force_redownload: Ignore cache and redownload
        :return: Path to dataset directory
        """
        target_dir = self.project_root / "data" / name
        
        # Return if valid and not forcing redownload
        if not force_redownload and self._is_valid(target_dir, name):
            return target_dir
            
        # Try cached version
        cached_file = self.cache_dir / f"{name}.zip"
        if cached_file.exists():
            self._extract_cached(cached_file, target_dir)
            if self._is_valid(target_dir, name):
                return target_dir
                
        # Download via API
        downloaded_path = self.downloader.download_dataset(name)
        self._cache_dataset(downloaded_path, name)
        return downloaded_path

    def _is_valid(self, path: Path, name: str) -> bool:
        """Check if dataset exists and passes integrity checks"""
        # Check required files
        required_files = self._get_required_files(name)
        if not all((path / f).exists() for f in required_files):
            return False
            
        # Check checksum if available
        if name in self.dataset_checksums:
            return self._verify_checksum(path, name)
            
        return True

    def _get_required_files(self, name: str) -> List[str]:
        """Get required files from config"""
        return {
            "IIIT5K": ["train.json", "test.json"],
            "SynthText": ["SynthText.h5"]
        }.get(name, [])

    def _verify_checksum(self, path: Path, name: str) -> bool:
        """Verify dataset integrity using SHA256 checksum"""
        expected_checksum = self.dataset_checksums[name]
        actual_checksum = self._calculate_checksum(path)
        return actual_checksum == expected_checksum

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum for a directory"""
        sha = hashlib.sha256()
        for file_path in sorted(path.glob('**/*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    while chunk := f.read(8192):
                        sha.update(chunk)
        return sha.hexdigest()

    def _cache_dataset(self, dataset_path: Path, name: str):
        """Create zip cache of downloaded dataset"""
        cached_file = self.cache_dir / f"{name}.zip"
        shutil.make_archive(str(cached_file.with_suffix('')), 'zip', dataset_path)
        
        # Update checksum
        self.dataset_checksums[name] = self._calculate_checksum(dataset_path)
        self._save_checksums()

    def _extract_cached(self, zip_path: Path, target_dir: Path):
        """Extract cached dataset to target directory"""
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)
        shutil.unpack_archive(str(zip_path), str(target_dir))

    def _load_checksums(self) -> Dict[str, str]:
        """Load checksums from file"""
        checksum_file = self.cache_dir / "checksums.json"
        if checksum_file.exists():
            import json
            return json.loads(checksum_file.read_text())
        return {}

    def _save_checksums(self):
        """Save updated checksums to file"""
        checksum_file = self.cache_dir / "checksums.json"
        import json
        checksum_file.write_text(json.dumps(self.dataset_checksums, indent=2))

# Create singleton instance
dataset_manager = AutoDataset()