# src/model/crnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) for text recognition
    Architecture: CNN Feature Extractor + RNN Sequence Modeling + CTC
    """
    
    def __init__(self, config):
        super(CRNN, self).__init__()
        self.config = config
        
        # CNN Feature Extractor
        self.cnn = self._build_cnn()
        
        # RNN Sequence Modeling
        self.rnn = self._build_rnn()
        
        # Classification Layer
        self.classifier = nn.Linear(
            self.config['model']['rnn']['hidden_size'] * 2 if self.config['model']['rnn']['bidirectional'] else self.config['model']['rnn']['hidden_size'],
            self.config['model']['num_classes']
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)
        
    def _build_cnn(self):
        """Build CNN feature extractor"""
        cnn_config = self.config['model']['cnn']
        
        if cnn_config['feature_extractor'] == 'resnet':
            # Use ResNet-18 as feature extractor
            resnet = models.resnet18(pretrained=cnn_config['pretrained'])
            
            # Remove the last two layers (avgpool and fc)
            layers = list(resnet.children())[:-2]
            
            # Modify first conv layer for grayscale if needed
            if self.config['model']['input_channels'] == 1:
                layers[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            cnn = nn.Sequential(*layers)
            
            # Add adaptive pooling to ensure consistent output size
            cnn.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, None)))
            
        elif cnn_config['feature_extractor'] == 'vgg':
            # Use VGG-16 as feature extractor
            vgg = models.vgg16(pretrained=cnn_config['pretrained'])
            cnn = vgg.features
            
            # Add adaptive pooling
            cnn.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, None)))
            
        else:
            # Custom CNN architecture
            cnn = nn.Sequential(
                # First conv block
                nn.Conv2d(self.config['model']['input_channels'], 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                # Second conv block
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                # Third conv block
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                # Fourth conv block
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                # Fifth conv block
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                # Adaptive pooling to height 1
                nn.AdaptiveAvgPool2d((1, None))
            )
            
        return cnn
    
    def _build_rnn(self):
        """Build RNN sequence modeling layer"""
        rnn_config = self.config['model']['rnn']
        
        # Determine input size based on CNN output
        if self.config['model']['cnn']['feature_extractor'] == 'resnet':
            input_size = 512  # ResNet-18 output channels
        elif self.config['model']['cnn']['feature_extractor'] == 'vgg':
            input_size = 512  # VGG-16 output channels
        else:
            input_size = 512  # Custom CNN output channels
            
        if rnn_config['type'] == 'LSTM':
            rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=rnn_config['hidden_size'],
                num_layers=rnn_config['num_layers'],
                dropout=rnn_config['dropout'] if rnn_config['num_layers'] > 1 else 0,
                bidirectional=rnn_config['bidirectional'],
                batch_first=True
            )
        elif rnn_config['type'] == 'GRU':
            rnn = nn.GRU(
                input_size=input_size,
                hidden_size=rnn_config['hidden_size'],
                num_layers=rnn_config['num_layers'],
                dropout=rnn_config['dropout'] if rnn_config['num_layers'] > 1 else 0,
                bidirectional=rnn_config['bidirectional'],
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_config['type']}")
            
        return rnn
    
    def forward(self, x):
        """Forward pass"""
        batch_size = x.size(0)
        
        # CNN feature extraction
        conv_features = self.cnn(x)  # Shape: (batch, channels, 1, width)
        
        # Reshape for RNN input
        conv_features = conv_features.squeeze(2)  # Shape: (batch, channels, width)
        conv_features = conv_features.permute(0, 2, 1)  # Shape: (batch, width, channels)
        
        # RNN sequence modeling
        rnn_output, _ = self.rnn(conv_features)  # Shape: (batch, seq_len, hidden_size)
        
        # Apply dropout
        rnn_output = self.dropout(rnn_output)
        
        # Classification
        output = self.classifier(rnn_output)  # Shape: (batch, seq_len, num_classes)
        
        # Apply log softmax for CTC loss
        output = F.log_softmax(output, dim=2)
        
        return output


class CTCLoss(nn.Module):
    """CTC Loss for sequence labeling"""
    
    def __init__(self, blank_idx=0):
        super(CTCLoss, self).__init__()
        self.blank_idx = blank_idx
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
        
    def forward(self, predictions, targets, pred_lengths, target_lengths):
        """
        Args:
            predictions: (batch_size, seq_len, num_classes)
            targets: (batch_size, target_len)
            pred_lengths: (batch_size,)
            target_lengths: (batch_size,)
        """
        # CTC expects (seq_len, batch_size, num_classes)
        predictions = predictions.permute(1, 0, 2)
        
        # Flatten targets
        targets = targets.flatten()
        
        # Calculate CTC loss
        loss = self.ctc_loss(predictions, targets, pred_lengths, target_lengths)
        
        return loss


class CRNNWithAttention(CRNN):
    """CRNN with attention mechanism"""
    
    def __init__(self, config):
        super(CRNNWithAttention, self).__init__(config)
        
        if config['model']['use_attention']:
            self.attention = nn.MultiheadAttention(
                embed_dim=config['model']['rnn']['hidden_size'] * 2 if config['model']['rnn']['bidirectional'] else config['model']['rnn']['hidden_size'],
                num_heads=8,
                dropout=0.1
            )
    
    def forward(self, x):
        """Forward pass with attention"""
        batch_size = x.size(0)
        
        # CNN feature extraction
        conv_features = self.cnn(x)
        conv_features = conv_features.squeeze(2)
        conv_features = conv_features.permute(0, 2, 1)
        
        # RNN sequence modeling
        rnn_output, _ = self.rnn(conv_features)
        
        # Apply attention if enabled
        if self.config['model']['use_attention']:
            # Self-attention
            attended_output, _ = self.attention(rnn_output, rnn_output, rnn_output)
            rnn_output = attended_output + rnn_output  # Residual connection
        
        # Apply dropout
        rnn_output = self.dropout(rnn_output)
        
        # Classification
        output = self.classifier(rnn_output)
        output = F.log_softmax(output, dim=2)
        
        return output


def create_model(config):
    """Factory function to create CRNN model"""
    if config['model']['use_attention']:
        return CRNNWithAttention(config)
    else:
        return CRNN(config)


# Model utilities
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model):
    """Initialize model weights"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)


if __name__ == "__main__":
    # Test the model
    import yaml
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config)
    
    # Print model info
    print(f"Model: {model}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    height = config['model']['input_height']
    width = config['model']['input_width']
    channels = config['model']['input_channels']
    
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")