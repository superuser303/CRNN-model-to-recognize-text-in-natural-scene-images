#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_path = project_root / '.env'
if env_path.exists():
    print(f"Loading environment variables from {env_path}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip('\'"')
    print("Environment variables loaded successfully")
else:
    print(f"Warning: .env file not found at {env_path}")

# Ensure config directory exists
config_dir = project_root / "configs"
config_dir.mkdir(exist_ok=True)

# Create default config if missing
config_file = config_dir / "datasets.yaml"
if not config_file.exists():
    config_file.write_text("""datasets:
  IIIT5K:
    required_files: ["train.json", "test.json"]
    sources:
      - name: "Roboflow Mirror"
        type: "roboflow"
        url: "https://universe.roboflow.com/dod/iiit5k-lzg9f/download/yolov8"
        
      - name: "GitHub Mirror"
        type: "direct"
        url: "https://github.com/adumrewal/iiit-5k-word-coco-dataset/archive/refs/heads/main.zip"
        
      - name: "Robofallback"
        type: "direct"
        url: "https://github.com/superuser303/IIIT5K-Dataset/releases/download/v1.0/IIIT5K.zip"

  SynthText:
    required_files: ["SynthText.h5"]
    sources:
      - name: "Official Source"
        type: "direct"
        url: "https://thor.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip"
        
      - name: "Mirror Source"
        type: "direct"
        url: "https://github.com/superuser303/SynthText-Dataset/releases/download/v1.0/SynthText.zip"
""")
    print(f"Created default config at: {config_file}")

from data import dataset_manager

if __name__ == "__main__":
    # Verify Roboflow API key is loaded
    roboflow_key = os.getenv("ROBOFLOW_API_KEY")
    if roboflow_key:
        print(f"Using Roboflow API key: {roboflow_key[:4]}...{roboflow_key[-4:]}")
    else:
        print("ROBOFLOW_API_KEY not set, using fallback sources")
    
    print("Downloading IIIT5K dataset...")
    iiit_path = dataset_manager.get_dataset("IIIT5K")
    print(f"IIIT5K dataset ready at: {iiit_path}")
    
    print("\nDownloading SynthText dataset...")
    synth_path = dataset_manager.get_dataset("SynthText")
    print(f"SynthText dataset ready at: {synth_path}")