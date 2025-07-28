"""Autoencoder model implementation for anomaly detection."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from ..utils.logger import LoggerMixin
try:
    from ..utils.config import Config
except ImportError:
    # Fallback for when pydantic is not available
    try:
        from ..utils.config_manager import SimpleConfigManager as Config
    except ImportError:
        # Final fallback
        class Config:
            pass


class AutoencoderModel(nn.Module, LoggerMixin):
    """Simple autoencoder model for network anomaly detection."""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 2,
                 activation: str = "relu",
                 dropout_rate: float = 0.1,
                 batch_norm: bool = False):
        """Initialize autoencoder model.
        
        Args:
            input_size: Number of input features.
            hidden_size: Size of the hidden layer (bottleneck).
            activation: Activation function ('relu', 'tanh', 'sigmoid').
            dropout_rate: Dropout rate for regularization.
            batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        LoggerMixin.__init__(self)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Activation function
        if activation.lower() == "relu":
            self.act_fn = nn.ReLU()
        elif activation.lower() == "tanh":
            self.act_fn = nn.Tanh()
        elif activation.lower() == "sigmoid":
            self.act_fn = nn.Sigmoid()
        else:
            self.logger.warning(f"Unknown activation: {activation}. Using ReLU.")
            self.act_fn = nn.ReLU()
        
        # Encoder layers
        self.encoder = nn.Sequential()
        self.encoder.add_module("linear1", nn.Linear(input_size, hidden_size))
        
        if batch_norm:
            self.encoder.add_module("batchnorm1", nn.BatchNorm1d(hidden_size))
        
        self.encoder.add_module("activation1", self.act_fn)
        
        if dropout_rate > 0:
            self.encoder.add_module("dropout1", nn.Dropout(dropout_rate))
        
        # Decoder layers
        self.decoder = nn.Sequential()
        self.decoder.add_module("linear2", nn.Linear(hidden_size, input_size))
        
        # Initialize weights
        self._init_weights()
        
        self.logger.info(f"Autoencoder initialized: {input_size} -> {hidden_size} -> {input_size}")
    
    def _init_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_size).
            
        Returns:
            Reconstructed tensor of shape (batch_size, input_size).
        """
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, input_size).
            
        Returns:
            Encoded tensor of shape (batch_size, hidden_size).
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to reconstruction.
        
        Args:
            z: Latent tensor of shape (batch_size, hidden_size).
            
        Returns:
            Decoded tensor of shape (batch_size, input_size).
        """
        return self.decoder(z)
    
    def reconstruction_error(self, x: torch.Tensor, reduction: str = "none") -> torch.Tensor:
        """Calculate reconstruction error for input samples.
        
        Args:
            x: Input tensor of shape (batch_size, input_size).
            reduction: How to reduce the error ('none', 'mean', 'sum').
            
        Returns:
            Reconstruction error tensor.
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = F.mse_loss(reconstructed, x, reduction=reduction)
            
            if reduction == "none":
                # Return per-sample error (mean across features)
                error = error.mean(dim=1)
                
        return error
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics.
        
        Returns:
            Dictionary with model information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "Autoencoder",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "batch_norm": self.batch_norm,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 ** 2)  # Approximate size in MB
        }
    
    def save_model(self, save_path: Path, include_config: bool = True) -> None:
        """Save model state and configuration.
        
        Args:
            save_path: Path to save the model.
            include_config: Whether to save model configuration.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "state_dict": self.state_dict(),
            "model_info": self.get_model_info()
        }
        
        if include_config:
            save_data["config"] = {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "batch_norm": self.batch_norm
            }
        
        torch.save(save_data, save_path)
        self.logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, model_path: Path, device: Optional[torch.device] = None) -> "AutoencoderModel":
        """Load model from saved state.
        
        Args:
            model_path: Path to the saved model.
            device: Device to load the model on.
            
        Returns:
            Loaded model instance.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration
        config = checkpoint.get("config", {})
        
        # Create model instance
        model = cls(
            input_size=config.get("input_size", 4),
            hidden_size=config.get("hidden_size", 2),
            activation=config.get("activation", "relu"),
            dropout_rate=config.get("dropout_rate", 0.1),
            batch_norm=config.get("batch_norm", False)
        )
        
        # Load state dict
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        
        logger = model.logger
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model info: {model.get_model_info()}")
        
        return model


class SimpleNumpyAutoencoder:
    """Simple NumPy-based autoencoder for cases where PyTorch is not available."""
    
    def __init__(self, input_size: int, hidden_size: int = 2):
        """Initialize NumPy autoencoder.
        
        Args:
            input_size: Number of input features.
            hidden_size: Size of hidden layer.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, input_size) * 0.1
        self.b2 = np.zeros((1, input_size))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU activation."""
        return (x > 0).astype(float)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the autoencoder.
        
        Args:
            X: Input array of shape (batch_size, input_size).
            
        Returns:
            Reconstructed array of shape (batch_size, input_size).
        """
        # Encoder
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Decoder
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.z2
        
        return self.output
    
    def compute_loss(self, X: np.ndarray, X_pred: np.ndarray) -> float:
        """Compute MSE loss.
        
        Args:
            X: Original input.
            X_pred: Predicted output.
            
        Returns:
            Mean squared error.
        """
        return np.mean((X - X_pred) ** 2)
    
    def backward(self, X: np.ndarray, learning_rate: float = 0.001) -> None:
        """Backward pass for training.
        
        Args:
            X: Input array.
            learning_rate: Learning rate for updates.
        """
        m = X.shape[0]
        
        # Compute gradients
        dL_dz2 = 2 * (self.output - X) / m
        dL_dW2 = np.dot(self.a1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * self.relu_derivative(self.z1)
        dL_dW1 = np.dot(X.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
    
    def train(self, X: np.ndarray, epochs: int = 100, learning_rate: float = 0.001, 
              batch_size: int = 32, verbose: bool = True) -> list:
        """Train the autoencoder.
        
        Args:
            X: Training data.
            epochs: Number of training epochs.
            learning_rate: Learning rate.
            batch_size: Batch size.
            verbose: Whether to print progress.
            
        Returns:
            List of training losses.
        """
        losses = []
        n_batches = len(X) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                
                # Forward pass
                X_pred = self.forward(X_batch)
                loss = self.compute_loss(X_batch, X_pred)
                epoch_loss += loss
                
                # Backward pass
                self.backward(X_batch, learning_rate)
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:3d}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction for input data.
        
        Args:
            X: Input data.
            
        Returns:
            Reconstructed data.
        """
        return self.forward(X)
    
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Calculate reconstruction error per sample.
        
        Args:
            X: Input data.
            
        Returns:
            Reconstruction errors.
        """
        X_pred = self.predict(X)
        errors = np.mean((X - X_pred) ** 2, axis=1)
        return errors
