from pathlib import Path
from .api_downloader import APIDatasetDownloader

class AutoDataset:
    def __init__(self):
        self.downloader = APIDatasetDownloader()
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(exist_ok=True)

    def get_dataset(self, name, force=False):
        """Automatically handles dataset availability"""
        target_dir = Path("data") / name
        
        # Check existing
        if not force and self._is_valid(target_dir, name):
            return target_dir
            
        # Download via API
        return self.downloader.download_dataset(name)

    def _is_valid(self, path, name):
        """Check required files"""
        required = {
            "IIIT5K": ["train.json", "test.json"],
            "SynthText": ["SynthText.h5"]
        }.get(name, [])
        return all((path / f).exists() for f in required)

# Singleton instance
dataset_manager = AutoDataset()