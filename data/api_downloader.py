import os
import requests
import zipfile
import yaml
from pathlib import Path
import time
import shutil

class APIDatasetDownloader:
    def __init__(self, config_path: str = None):
        # Get the project root directory
        self.project_root = Path(__file__).resolve().parent.parent
        
        # Set default config path if not provided
        if config_path is None:
            self.config_path = self.project_root / "configs" / "datasets.yaml"
        else:
            self.config_path = self.project_root / config_path
            
        self.config = self._load_config()
        self.cache_dir = self.project_root / "data" / "cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _load_config(self):
        """Load config from absolute path"""
        if not self.config_path.exists():
            # Try to find the config file in the standard location
            fallback_path = self.project_root / "configs" / "datasets.yaml"
            if fallback_path.exists():
                self.config_path = fallback_path
            else:
                raise FileNotFoundError(f"Config file not found at: {self.config_path}")
            
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def download_dataset(self, name):
        """Main interface for dataset download"""
        if name not in self.config["datasets"]:
            raise ValueError(f"Unknown dataset: {name}")
            
        dataset_cfg = self.config["datasets"][name]
        target_dir = self.project_root / "data" / name
        
        # Check existing dataset
        if self._is_valid(target_dir, dataset_cfg.get("required_files", [])):
            return target_dir
            
        # Try all sources
        for source in dataset_cfg["sources"]:
            try:
                print(f"Trying source: {source['name']}")
                if source["type"] == "direct":
                    return self._download_direct(source, target_dir)
                elif source["type"] == "roboflow":
                    # Use environment variable directly
                    api_key = os.getenv("ROBOFLOW_API_KEY")
                    if not api_key:
                        print("ROBOFLOW_API_KEY not set, skipping Roboflow source")
                        continue
                    return self._download_roboflow(source, target_dir, api_key)
            except Exception as e:
                print(f"Download failed: {str(e)}")
                continue
                
        raise RuntimeError(f"All download attempts failed for {name}")

    def _download_direct(self, source, target_dir):
        """Handle direct downloads"""
        response = requests.get(source["url"], stream=True)
        response.raise_for_status()
        
        # Stream download with progress
        temp_file = self.cache_dir / f"temp_{time.time_ns()}.zip"
        try:
            with open(temp_file, "wb") as f:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                start_time = time.time()
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Show progress
                        if total_size > 0:
                            percent = 100 * downloaded / total_size
                            speed = downloaded / (time.time() - start_time) / 1024
                            print(f"\rProgress: {percent:.1f}% ({downloaded/1024/1024:.1f}MB) Speed: {speed:.1f}KB/s", end="")
                
                print()  # New line after progress
                
            # Extract and cleanup
            print(f"Extracting to {target_dir}")
            with zipfile.ZipFile(temp_file) as zip_ref:
                zip_ref.extractall(target_dir)
            return target_dir
        finally:
            # Cleanup temp file
            if temp_file.exists():
                temp_file.unlink()

    def _download_roboflow(self, source, target_dir, api_key):
        """Handle Roboflow API downloads"""
        url = f"{source['url']}?api_key={api_key}"
        print(f"Using Roboflow URL: {url.split('?')[0]}...")  # Don't show full API key
        return self._download_direct({"url": url, "type": "direct"}, target_dir)

    def _is_valid(self, path, required_files):
        """Validate dataset integrity"""
        return all((path / f).exists() for f in required_files)