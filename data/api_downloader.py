import os
import requests
import zipfile
import io
import yaml
from pathlib import Path
from tqdm import tqdm
import hashlib
import time
# data/api_downloader.py
api_key = os.getenv("ROBOFLOW_API_KEY") or self.config["datasets"]["IIIT5K"]["sources"][0].get("api_key", "")
class APIDatasetDownloader:
    def __init__(self, config_path="configs/datasets.yaml"):
        self.config = self._load_config(config_path)
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    def download_dataset(self, name):
        """Main interface for dataset download"""
        if name not in self.config["datasets"]:
            raise ValueError(f"Unknown dataset: {name}")
            
        dataset_cfg = self.config["datasets"][name]
        target_dir = Path("data") / name
        
        # Check existing dataset
        if self._is_valid(target_dir, dataset_cfg.get("required_files", [])):
            return target_dir
            
        # Try all sources
        for source in dataset_cfg["sources"]:
            try:
                if source["type"] == "direct":
                    return self._download_direct(source, target_dir)
                elif source["type"] == "roboflow":
                    return self._download_roboflow(source, target_dir)
            except Exception as e:
                print(f"Download failed: {e}")
                continue
                
        raise RuntimeError(f"All download attempts failed for {name}")

    def _download_direct(self, source, target_dir):
        """Handle direct downloads"""
        response = requests.get(source["url"], stream=True)
        response.raise_for_status()
        
        # Stream download with progress
        temp_file = self.cache_dir / f"temp_{time.time_ns()}.zip"
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # Extract and cleanup
        with zipfile.ZipFile(temp_file) as zip_ref:
            zip_ref.extractall(target_dir)
        temp_file.unlink()
        
        return target_dir

    def _download_roboflow(self, source, target_dir):
        """Handle Roboflow API downloads"""
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY not set")
            
        url = f"{source['url']}?api_key={api_key}"
        return self._download_direct({"url": url, "type": "direct"}, target_dir)

    def _is_valid(self, path, required_files):
        """Validate dataset integrity"""
        return all((path / f).exists() for f in required_files)