"""Core training functionality for the autoencoder model."""

import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.logger import LoggerMixin
from ..utils.enterprise_config import ConfigurationManager
from ..data.processor import DataProcessor
from ..models.autoencoder import AutoencoderModel, SimpleNumpyAutoencoder


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            epoch: Current epoch number.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
            
        return False


class ModelTrainer(LoggerMixin):
    """Model trainer for autoencoder-based anomaly detection."""
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize model trainer.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__()
        self.config_manager = config_manager or ConfigurationManager()
        self.config = self.config_manager.config
        self.model = None
        self.optimizer = None
        self.device = None
        self.training_history = []
        
        # Initialize device
        self._setup_device()
        
        # Initialize model
        self._init_model()
    
    def _setup_device(self) -> None:
        """Setup compute device (CPU/GPU)."""
        if TORCH_AVAILABLE:
            device_config = self.config.training.device
            
            if device_config == 'auto':
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif device_config == 'cuda':
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    self.logger.warning("CUDA requested but not available. Using CPU.")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(device_config)
            
            self.logger.info(f"Using device: {self.device}")
        else:
            self.device = "cpu"
            self.logger.info("PyTorch not available. Using NumPy implementation.")
    
    def _init_model(self) -> None:
        """Initialize model based on configuration."""
        from ..models.autoencoder import AutoencoderModel
        
        self.model = AutoencoderModel(
            input_size=self.config.model.input_dim,
            hidden_size=2  # Default hidden size
        )
    
    def _create_data_loaders(self, 
                           train_data: np.ndarray, 
                           val_data: Optional[np.ndarray] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create PyTorch data loaders.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            
        Returns:
            Tuple of (train_loader, val_loader).
        """
        if not TORCH_AVAILABLE:
            return None, None
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_data).to(self.device)
        train_dataset = TensorDataset(train_tensor, train_tensor)  # Input = Target for autoencoder
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.training.shuffle,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False
        )
        
        val_loader = None
        if val_data is not None:
            val_tensor = torch.FloatTensor(val_data).to(self.device)
            val_dataset = TensorDataset(val_tensor, val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
        
        return train_loader, val_loader
    
    def train_pytorch_model(self, 
                          train_data: np.ndarray, 
                          val_data: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """Train PyTorch autoencoder model.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            
        Returns:
            Training history dictionary.
        """
        self.logger.info("Starting PyTorch model training...")
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(train_data, val_data)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [] if val_data is not None else None
        }
        
        # Early stopping
        early_stopping = None
        if self.config.training.early_stopping_enabled:
            early_stopping = EarlyStopping(
                patience=self.config.training.patience,
                min_delta=self.config.training.min_delta
            )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            epoch_start = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_data, batch_target in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(batch_data)
                loss = criterion(output, batch_target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_batches = 0
                    for batch_data, batch_target in val_loader:
                        output = self.model(batch_data)
                        loss = criterion(output, batch_target)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                history['val_loss'].append(avg_val_loss)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.model.state_dict().copy()
                
                # Check early stopping
                if early_stopping and early_stopping(avg_val_loss, epoch):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                avg_val_loss = avg_train_loss
                if avg_train_loss < best_val_loss:
                    best_val_loss = avg_train_loss
                    best_model_state = self.model.state_dict().copy()
            
            epoch_time = time.time() - epoch_start
            
            # Log progress
            if epoch % 20 == 0:
                if val_data is not None:
                    self.logger.info(
                        f"Epoch {epoch:3d}/{self.config.training.epochs} - "
                        f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} "
                        f"({epoch_time:.2f}s)"
                    )
                else:
                    self.logger.info(
                        f"Epoch {epoch:3d}/{self.config.training.epochs} - "
                        f"Train Loss: {avg_train_loss:.6f} ({epoch_time:.2f}s)"
                    )
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        return history
    
    def train_numpy_model(self, 
                         train_data: np.ndarray, 
                         val_data: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """Train NumPy autoencoder model.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            
        Returns:
            Training history dictionary.
        """
        self.logger.info("Starting NumPy model training...")
        
        start_time = time.time()
        
        # Train the model
        losses = self.model.train(
            train_data,
            epochs=self.config.training.epochs,
            learning_rate=self.config.training.learning_rate,
            batch_size=self.config.training.batch_size,
            verbose=False
        )
        
        # Calculate validation loss if provided
        val_losses = []
        if val_data is not None:
            for i in range(0, len(losses), 20):  # Check every 20 epochs
                val_pred = self.model.predict(val_data)
                val_loss = self.model.compute_loss(val_data, val_pred)
                val_losses.append(val_loss)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Final training loss: {losses[-1]:.6f}")
        
        history = {
            'train_loss': losses,
            'val_loss': val_losses if val_losses else None
        }
        
        return history
    
    def train(self, 
              train_data: np.ndarray, 
              val_data: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """Train the autoencoder model.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            
        Returns:
            Training history dictionary.
        """
        self.log_method_call("train", train_shape=train_data.shape, 
                           val_shape=val_data.shape if val_data is not None else None)
        
        start_time = time.time()
        
        if TORCH_AVAILABLE:
            history = self.train_pytorch_model(train_data, val_data)
        else:
            history = self.train_numpy_model(train_data, val_data)
        
        self.training_history = history
        
        # Log performance
        self.log_performance("model_training", time.time() - start_time,
                           final_train_loss=history['train_loss'][-1],
                           final_val_loss=history['val_loss'][-1] if history['val_loss'] else None)
        
        return history
    
    def save_model(self, model_path: Union[str, Path]) -> None:
        """Save the trained model.
        
        Args:
            model_path: Path to save the model.
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if TORCH_AVAILABLE and isinstance(self.model, AutoencoderModel):
            self.model.save_model(model_path)
        else:
            # Save NumPy model
            import pickle
            model_data = {
                'model': self.model,
                'config': self.config.to_dict(),
                'training_history': self.training_history
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load a trained model.
        
        Args:
            model_path: Path to the saved model.
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if TORCH_AVAILABLE:
            try:
                self.model = AutoencoderModel.load_model(model_path, self.device)
                self.logger.info(f"PyTorch model loaded from {model_path}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load as PyTorch model: {e}")
        
        # Try to load as NumPy model
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            if 'training_history' in model_data:
                self.training_history = model_data['training_history']
            
            self.logger.info(f"NumPy model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results.
        
        Returns:
            Dictionary with training summary.
        """
        if not self.training_history:
            return {"status": "No training history available"}
        
        summary = {
            "status": "completed",
            "epochs_trained": len(self.training_history['train_loss']),
            "final_train_loss": self.training_history['train_loss'][-1],
            "best_train_loss": min(self.training_history['train_loss']),
            "model_type": "PyTorch" if TORCH_AVAILABLE and isinstance(self.model, AutoencoderModel) else "NumPy"
        }
        
        if self.training_history['val_loss']:
            summary.update({
                "final_val_loss": self.training_history['val_loss'][-1],
                "best_val_loss": min(self.training_history['val_loss'])
            })
        
        if TORCH_AVAILABLE and isinstance(self.model, AutoencoderModel):
            summary.update(self.model.get_model_info())
        
        return summary
