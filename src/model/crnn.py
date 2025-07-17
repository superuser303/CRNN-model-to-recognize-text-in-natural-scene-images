"""
CRNN Model Implementation for Text Recognition
Combines CNN for feature extraction with RNN for sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM layer for sequence modeling"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, seq_len, output_size)
        """
        recurrent, _ = self.rnn(input_tensor)
        output = self.linear(recurrent)
        return output


class CNNFeatureExtractor(nn.Module):
    """CNN backbone for feature extraction"""
    
    def __init__(self, input_channels=3, feature_extractor='resnet', pretrained=True):
        super(CNNFeatureExtractor, self).__init__()
        self.feature_extractor = feature_extractor
        
        if feature_extractor == 'resnet':
            self.cnn = self._build_resnet_backbone(input_channels, pretrained)
        elif feature_extractor == 'vgg':
            self.cnn = self._build_vgg_backbone(input_channels, pretrained)
        else:
            self.cnn = self._build_custom_backbone(input_channels)
    
    def _build_resnet_backbone(self, input_channels, pretrained):
        """Build ResNet-based feature extractor"""
        resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer if input channels != 3
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove fully connected layers
        modules = list(resnet.children())[:-2]
        cnn = nn.Sequential(*modules)
        
        # Add adaptive pooling to ensure consistent output size
        cnn.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, None)))
        
        return cnn
    
    def _build_vgg_backbone(self, input_channels, pretrained):
        """Build VGG-based feature extractor"""
        vgg = models.vgg16_bn(pretrained=pretrained)
        
        # Modify first conv layer if input channels != 3
        if input_channels != 3:
            vgg.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        
        # Use only feature extraction part
        cnn = vgg.features
        cnn.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, None)))
        
        return cnn
    
    def _build_custom_backbone(self, input_channels):
        """Build custom CNN backbone"""
        return nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            
            # Conv Block 4
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Conv Block 5
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            
            # Conv Block 6
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Final conv layer
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            features: (batch_size, feature_dim, seq_len)
        """
        conv_features = self.cnn(x)  # (batch_size, channels, height, width)
        
        # Reshape for RNN: (batch_size, channels, height, width) -> (batch_size, seq_len, feature_dim)
        batch_size, channels, height, width = conv_features.size()
        
        if height != 1:
            # If height > 1, apply global average pooling
            conv_features = F.adaptive_avg_pool2d(conv_features, (1, width))
            height = 1
        
        # Reshape to (batch_size, seq_len, feature_dim)
        conv_features = conv_features.view(batch_size, channels * height, width)
        conv_features = conv_features.permute(0, 2, 1)  # (batch_size, width, channels)
        
        return conv_features


class CRNN(nn.Module):
    """
    Complete CRNN model for text recognition
    Architecture: CNN Feature Extractor -> RNN -> Output Layer
    """
    
    def __init__(self, config):
        super(CRNN, self).__init__()
        
        # Extract configuration
        self.input_height = config.model.input_height
        self.input_width = config.model.input_width
        self.input_channels = config.model.input_channels
        self.num_classes = config.model.num_classes
        
        # CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(
            input_channels=self.input_channels,
            feature_extractor=config.model.cnn.feature_extractor,
            pretrained=config.model.cnn.pretrained
        )
        
        # Calculate RNN input size based on CNN output
        self.rnn_input_size = self._get_rnn_input_size(config)
        
        # RNN layers
        self.rnn_type = config.model.rnn.type
        self.hidden_size = config.model.rnn.hidden_size
        self.num_layers = config.model.rnn.num_layers
        
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.rnn_input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=config.model.rnn.bidirectional,
                dropout=config.model.rnn.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.rnn_input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=config.model.rnn.bidirectional,
                dropout=config.model.rnn.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type == 'BiLSTM':
            self.rnn = BidirectionalLSTM(
                self.rnn_input_size,
                self.hidden_size,
                self.hidden_size * 2
            )
        
        # Output layer
        rnn_output_size = self.hidden_size * 2 if config.model.rnn.bidirectional else self.hidden_size
        self.classifier = nn.Linear(rnn_output_size, self.num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.model.rnn.dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_rnn_input_size(self, config):
        """Calculate RNN input size based on CNN architecture"""
        if config.model.cnn.feature_extractor == 'resnet':
            return 512
        elif config.model.cnn.feature_extractor == 'vgg':
            return 512
        else:
            return 512  # Custom backbone output size
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
    
    def forward(self, x):
        """
        Forward pass through the CRNN model
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            output: Tensor of shape (batch_size, seq_len, num_classes)
        """
        # CNN feature extraction
        conv_features = self.cnn(x)  # (batch_size, seq_len, feature_dim)
        
        # RNN sequence modeling
        if self.rnn_type in ['LSTM', 'GRU']:
            rnn_output, _ = self.rnn(conv_features)
        else:  # BiLSTM
            rnn_output = self.rnn(conv_features)
        
        # Apply dropout
        rnn_output = self.dropout(rnn_output)
        
        # Classification
        output = self.classifier(rnn_output)  # (batch_size, seq_len, num_classes)
        
        # Apply log softmax for CTC loss
        output = F.log_softmax(output, dim=2)
        
        return output
    
    def predict(self, x):
        """
        Prediction method for inference
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted character probabilities
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            # Apply softmax for probabilities
            probs = F.softmax(output, dim=2)
            return probs


def create_model(config):
    """
    Factory function to create CRNN model
    
    Args:
        config: Configuration object
        
    Returns:
        CRNN model instance
    """
    model = CRNN(config)
    return model


if __name__ == "__main__":
    # Test model creation
    from omegaconf import OmegaConf
    
    # Load default config
    config = OmegaConf.load("configs/config.yaml")
    
    # Create model
    model = create_model(config)
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 32, 128)
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model created successfully!")