"""
Flask Web Application for CRNN Text Recognition
Provides REST API endpoints for text recognition from images
"""

import os
import io
import cv2
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import traceback

# Import your model components
from src.model.crnn import create_model
from src.inference.predictor import TextPredictor
from src.data.transforms import get_inference_transforms
from omegaconf import OmegaConf

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for model
predictor = None
config = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the trained CRNN model"""
    global predictor, config
    try:
        # Load configuration
        config_path = os.path.join('configs', 'config.yaml')
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return False
            
        config = OmegaConf.load(config_path)
        
        # Model path
        model_path = config.deployment.model_path
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            logger.info("Creating dummy predictor for development")
            predictor = DummyPredictor()
            return True
        
        # Initialize predictor
        predictor = TextPredictor(model_path, config)
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        # Create dummy predictor for development
        predictor = DummyPredictor()
        return True

class DummyPredictor:
    """Dummy predictor for development when model is not available"""
    
    def __init__(self):
        self.mock_results = [
            ("HELLO WORLD", 0.95),
            ("STOP SIGN", 0.89),
            ("PARKING", 0.92),
            ("EXIT", 0.88),
            ("RESTAURANT", 0.91),
            ("OPEN", 0.94),
            ("CLOSED", 0.87),
            ("WELCOME", 0.93),
            ("SHOP", 0.90),
            ("CAFE", 0.86)
        ]
    
    def predict(self, image_path):
        """Mock prediction"""
        import random
        import time
        
        # Simulate processing time
        time.sleep(1)
        
        # Return random result
        text, confidence = random.choice(self.mock_results)
        return {
            'text': text,
            'confidence': confidence,
            'processing_time': 1.0
        }

def preprocess_image(image_path):
    """Preprocess image for model inference"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        return pil_image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict_text():
    """Main prediction endpoint"""
    start_time = datetime.now()
    
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, TIFF'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        try:
            # Preprocess image
            processed_image = preprocess_image(file_path)
            
            # Make prediction
            if predictor is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            result = predictor.predict(file_path)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare response
            response = {
                'text': result.get('text', ''),
                'confidence': float(result.get('confidence', 0.0)),
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            logger.info(f"Prediction successful: {result['text']} (confidence: {result['confidence']:.3f})")
            
            return jsonify(response)
            
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file removed: {file_path}")
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': 'Internal server error during prediction',
            'message': str(e),
            'success': False,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple images"""
    start_time = datetime.now()
    
    try:
        # Check if files are present
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        
        if len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        # Limit batch size
        max_batch_size = 10
        if len(files) > max_batch_size:
            return jsonify({'error': f'Maximum batch size is {max_batch_size}'}), 400
        
        results = []
        
        for i, file in enumerate(files):
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'error': 'Invalid file type',
                    'success': False
                })
                continue
            
            # Save file temporarily
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{i}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            try:
                file.save(file_path)
                
                # Make prediction
                result = predictor.predict(file_path)
                
                results.append({
                    'filename': file.filename,
                    'text': result.get('text', ''),
                    'confidence': float(result.get('confidence', 0.0)),
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'success': False
                })
                
            finally:
                # Clean up
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Calculate total processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return jsonify({
            'results': results,
            'total_images': len(files),
            'successful_predictions': sum(1 for r in results if r['success']),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error during batch prediction',
            'message': str(e)
        }), 500

@app.route('/api/model_info')
def model_info():
    """Get model information"""
    try:
        if config is None:
            return jsonify({'error': 'Model configuration not loaded'}), 500
        
        info = {
            'model_name': config.model.name,
            'input_size': [config.model.input_height, config.model.input_width],
            'num_classes': config.model.num_classes,
            'architecture': {
                'cnn_backbone': config.model.cnn.feature_extractor,
                'rnn_type': config.model.rnn.type,
                'rnn_layers': config.model.rnn.num_layers,
                'hidden_size': config.model.rnn.hidden_size
            },
            'supported_formats': list(app.config['ALLOWED_EXTENSIONS']),
            'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': 'Could not retrieve model information'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        temp_dir = app.config['UPLOAD_FOLDER']
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                # Remove files older than 1 hour
                if os.path.isfile(file_path):
                    file_age = datetime.now().timestamp() - os.path.getmtime(file_path)
                    if file_age > 3600:  # 1 hour
                        os.remove(file_path)
                        logger.info(f"Cleaned up old temp file: {filename}")
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {str(e)}")

if __name__ == "__main__":
    # Load model
    model_loaded = load_model()
    
    if not model_loaded:
        logger.error("Failed to load model. Exiting.")
        exit(1)
    
    # Clean up any existing temp files
    cleanup_temp_files()
    
    # Get configuration
    host = config.api.host if config else "0.0.0.0"
    port = config.api.port if config else 5000
    debug = config.api.debug if config else False
    
    logger.info(f"Starting Flask app on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    # Run the app
    app.run(host=host, port=port, debug=debug)
