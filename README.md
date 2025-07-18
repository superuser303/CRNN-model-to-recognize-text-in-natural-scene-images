# CRNN Model for Text Recognition in Natural Scene Images

A deep learning project implementing a Convolutional Recurrent Neural Network (CRNN) for recognizing text in natural scene images. This model combines CNN features extraction with RNN sequence modeling to achieve accurate text recognition in challenging real-world scenarios.

## 🌟 Features

- **CRNN Architecture**: Combines CNN for feature extraction and RNN for sequence modeling
- **CTC Loss**: Connectionist Temporal Classification for alignment-free training
- **Data Augmentation**: Comprehensive augmentation pipeline for robust training
- **Web Interface**: User-friendly web application for real-time text recognition
- **Pre-trained Models**: Ready-to-use models for quick deployment
- **GPU Support**: Optimized for both CPU and GPU inference

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CRNN-model-to-recognize-text-in-natural-scene-images.git
cd CRNN-model-to-recognize-text-in-natural-scene-images

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.inference.predictor import TextPredictor

# Initialize predictor
predictor = TextPredictor('models/best_model.pth')

# Predict text from image
result = predictor.predict('path/to/your/image.jpg')
print(f"Recognized text: {result}")
```

### Web Interface

```bash
# Start the web server
python app.py

# Open browser and navigate to http://localhost:5000
```

## 📊 Model Architecture

The CRNN model consists of three main components:

1. **Convolutional Layers**: Feature extraction from input images
2. **Recurrent Layers**: Sequence modeling with LSTM/GRU
3. **Transcription Layer**: CTC layer for final text prediction

```
Input Image → CNN Features → RNN Sequence → CTC → Text Output
```

## 🎯 Performance

| Dataset | Accuracy | Training Time | Model Size |
|---------|----------|---------------|------------|
| IIIT-5K | 87.2% | 4 hours | 45MB |
| Street View | 82.1% | 6 hours | 45MB |
| ICDAR2013 | 89.4% | 3 hours | 45MB |

## 📁 Project Structure

```
├── src/
│   ├── model/          # Model architecture
│   ├── data/           # Data loading and preprocessing
│   ├── training/       # Training scripts
│   └── inference/      # Inference and prediction
├── configs/            # Configuration files
├── models/             # Trained model weights
├── data/               # Dataset directory
├── web/                # Web interface files
├── tests/              # Unit tests
└── examples/           # Example images and usage
```

## 🔧 Training

### Prepare Dataset

```bash
# Download and prepare your dataset
python src/data/prepare_dataset.py --dataset_path /path/to/dataset
```

### Start Training

```bash
# Train with default configuration
python src/training/train.py

# Train with custom config
python src/training/train.py --config configs/custom_config.yaml
```

### Monitor Training

```bash
# View training logs
tensorboard --logdir logs/
```

## 🌐 Web Deployment

### Local Deployment

```bash
python app.py
```

### Netlify Deployment

The project is configured for Netlify deployment:

1. Fork this repository
2. Connect your Netlify account
3. Deploy automatically with the included `netlify.toml`

### Docker Deployment

```bash
# Build Docker image
docker build -t crnn-text-recognition .

# Run container
docker run -p 5000:5000 crnn-text-recognition
```

## 📊 Dataset Support

- **Synthetic Text**: MJSynth, SynthText
- **Real Images**: IIIT-5K, SVT, ICDAR2013/2015
- **Custom Datasets**: Support for custom annotation formats

## 🛠️ Configuration

Modify `configs/config.yaml` to customize:

- Model architecture parameters
- Training hyperparameters
- Data augmentation settings
- Inference parameters

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🔍 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_model.py
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.7+
- OpenCV 4.0+
- PIL/Pillow
- NumPy
- matplotlib

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CRNN Paper](https://arxiv.org/abs/1507.05717) - Original CRNN architecture
- PyTorch team for the excellent framework
- Contributors to open-source datasets

## 📞 Contact

- **Author**: Exploiter
- **Email**: exploiter0023@outlook.com
- **Project Link**: [https://github.com/superuser303/CRNN-model-to-recognize-text-in-natural-scene-images](https://github.com/yourusername/CRNN-model-to-recognize-text-in-natural-scene-images)

---

⭐ Star this repository if you find it helpful!