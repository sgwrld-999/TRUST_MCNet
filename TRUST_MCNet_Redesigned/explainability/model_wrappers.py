"""
Model wrappers for traditional ML models with SHAP integration.

This module provides wrappers for XGBoost, RandomForest, and IsolationForest
models to enable consistent SHAP-based explanations across different model types.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from abc import ABC, abstractmethod
import pickle
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLModelWrapper(ABC):
    """
    Abstract base class for ML model wrappers with SHAP integration.
    
    Provides consistent interface for training, prediction, and explanation
    across different model types.
    """
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        preprocessing: bool = True
    ):
        """
        Initialize model wrapper.
        
        Args:
            model_params: Parameters for the underlying model
            feature_names: Names of input features
            preprocessing: Whether to apply preprocessing
        """
        self.model_params = model_params or {}
        self.feature_names = feature_names or []
        self.preprocessing = preprocessing
        
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_metadata = {}
        
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance."""
        pass
    
    @abstractmethod
    def _get_model_type(self) -> str:
        """Get the model type identifier."""
        pass
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the model with optional validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics and metadata
        """
        logger.info(f"Training {self._get_model_type()} model...")
        
        # Apply preprocessing if enabled
        if self.preprocessing:
            X_train_processed, X_val_processed = self._preprocess_data(X_train, X_val)
        else:
            X_train_processed = X_train
            X_val_processed = X_val
        
        # Create and train model
        self.model = self._create_model()
        
        # Train the model
        training_result = self._train_model(X_train_processed, y_train, X_val_processed, y_val)
        
        self.is_trained = True
        self.training_metadata = training_result
        
        logger.info(f"Successfully trained {self._get_model_type()} model")
        return training_result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self._preprocess_features(X) if self.preprocessing else X
        return self._predict(X_processed)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Get prediction probabilities if supported.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities or None if not supported
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self._preprocess_features(X) if self.preprocessing else X
        return self._predict_proba(X_processed)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if supported.
        
        Returns:
            Feature importance scores or None if not supported
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return self._get_feature_importance()
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to file.
        
        Args:
            file_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'training_metadata': self.training_metadata,
            'model_type': self._get_model_type()
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved {self._get_model_type()} model to {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """
        Load a trained model from file.
        
        Args:
            file_path: Path to the saved model
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_params = model_data['model_params']
        self.training_metadata = model_data['training_metadata']
        self.is_trained = True
        
        logger.info(f"Loaded {self._get_model_type()} model from {file_path}")
    
    def _preprocess_data(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess training and validation data.
        
        Args:
            X_train: Training features
            X_val: Validation features
            
        Returns:
            Preprocessed training and validation features
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Sklearn not available, skipping preprocessing")
            return X_train, X_val
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform validation data if provided
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        return X_train_scaled, X_val_scaled
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess features using fitted scaler.
        
        Args:
            X: Input features
            
        Returns:
            Preprocessed features
        """
        if self.scaler is None:
            logger.warning("No scaler fitted, returning original features")
            return X
        
        return self.scaler.transform(X)
    
    @abstractmethod
    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train the underlying model."""
        pass
    
    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the underlying model."""
        pass
    
    def _predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities (default implementation)."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance (default implementation)."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


class XGBoostExplainer(MLModelWrapper):
    """
    XGBoost model wrapper with optimized SHAP TreeExplainer integration.
    
    Provides high-performance explanations for XGBoost models using
    the specialized TreeExplainer.
    """
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        preprocessing: bool = True
    ):
        """
        Initialize XGBoost explainer.
        
        Args:
            model_params: XGBoost-specific parameters
            feature_names: Names of input features
            preprocessing: Whether to apply preprocessing
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params, feature_names, preprocessing)
        
    def _create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier."""
        return xgb.XGBClassifier(**self.model_params)
    
    def _get_model_type(self) -> str:
        """Get model type identifier."""
        return "xgboost"
    
    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train XGBoost model with early stopping if validation data provided.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics
        """
        from datetime import datetime
        start_time = datetime.now()
        
        # Prepare evaluation set for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=10 if eval_set else None,
            verbose=False
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        train_accuracy = np.mean(y_train_pred == y_train)
        
        metrics = {
            'model_type': 'xgboost',
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'n_estimators_used': self.model.n_estimators,
            'best_iteration': getattr(self.model, 'best_iteration', None)
        }
        
        # Validation metrics if available
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            y_val_proba = self.model.predict_proba(X_val)[:, 1]
            
            metrics.update({
                'val_accuracy': np.mean(y_val_pred == y_val),
                'val_auc': roc_auc_score(y_val, y_val_proba) if SKLEARN_AVAILABLE else None
            })
        
        return metrics
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using XGBoost model."""
        return self.model.predict(X)
    
    def create_shap_explainer(self, background_data: Optional[np.ndarray] = None) -> 'shap.TreeExplainer':
        """
        Create optimized SHAP TreeExplainer for XGBoost.
        
        Args:
            background_data: Background data (not needed for TreeExplainer)
            
        Returns:
            SHAP TreeExplainer instance
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before creating explainer")
        
        # TreeExplainer is optimal for XGBoost
        explainer = shap.TreeExplainer(self.model)
        logger.info("Created SHAP TreeExplainer for XGBoost model")
        
        return explainer
    
    def explain_predictions(
        self,
        X: np.ndarray,
        explainer: Optional['shap.TreeExplainer'] = None,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions.
        
        Args:
            X: Input features to explain
            explainer: Pre-created SHAP explainer
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP explanation results
        """
        if explainer is None:
            explainer = self.create_shap_explainer()
        
        # Preprocess input if needed
        X_processed = self._preprocess_features(X) if self.preprocessing else X
        
        # Limit samples for performance
        if len(X_processed) > max_samples:
            indices = np.random.choice(len(X_processed), max_samples, replace=False)
            X_explain = X_processed[indices]
        else:
            X_explain = X_processed
            indices = np.arange(len(X_processed))
        
        # Generate SHAP values (fast for tree models)
        shap_values = explainer.shap_values(X_explain)
        
        # Get predictions
        predictions = self.predict(X_explain)
        
        # Calculate feature importance
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        return {
            'shap_values': shap_values,
            'predictions': predictions,
            'explained_indices': indices,
            'feature_importance': feature_importance,
            'base_value': explainer.expected_value,
            'model_type': 'xgboost'
        }


class RandomForestExplainer(MLModelWrapper):
    """
    Random Forest model wrapper with SHAP TreeExplainer integration.
    
    Provides efficient explanations for Random Forest models using
    the TreeExplainer optimized for ensemble methods.
    """
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        preprocessing: bool = True
    ):
        """
        Initialize Random Forest explainer.
        
        Args:
            model_params: Random Forest specific parameters
            feature_names: Names of input features
            preprocessing: Whether to apply preprocessing
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with: pip install scikit-learn")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params, feature_names, preprocessing)
    
    def _create_model(self) -> RandomForestClassifier:
        """Create Random Forest classifier."""
        return RandomForestClassifier(**self.model_params)
    
    def _get_model_type(self) -> str:
        """Get model type identifier."""
        return "randomforest"
    
    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics
        """
        from datetime import datetime
        start_time = datetime.now()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        train_accuracy = np.mean(y_train_pred == y_train)
        
        metrics = {
            'model_type': 'randomforest',
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth
        }
        
        # Validation metrics if available
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            y_val_proba = self.model.predict_proba(X_val)[:, 1]
            
            metrics.update({
                'val_accuracy': np.mean(y_val_pred == y_val),
                'val_auc': roc_auc_score(y_val, y_val_proba) if SKLEARN_AVAILABLE else None
            })
        
        return metrics
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using Random Forest model."""
        return self.model.predict(X)
    
    def create_shap_explainer(self, background_data: Optional[np.ndarray] = None) -> 'shap.TreeExplainer':
        """
        Create SHAP TreeExplainer for Random Forest.
        
        Args:
            background_data: Background data (not needed for TreeExplainer)
            
        Returns:
            SHAP TreeExplainer instance
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before creating explainer")
        
        # TreeExplainer works well with Random Forest
        explainer = shap.TreeExplainer(self.model)
        logger.info("Created SHAP TreeExplainer for Random Forest model")
        
        return explainer
    
    def explain_predictions(
        self,
        X: np.ndarray,
        explainer: Optional['shap.TreeExplainer'] = None,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for Random Forest predictions.
        
        Args:
            X: Input features to explain
            explainer: Pre-created SHAP explainer
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP explanation results
        """
        if explainer is None:
            explainer = self.create_shap_explainer()
        
        # Preprocess input if needed
        X_processed = self._preprocess_features(X) if self.preprocessing else X
        
        # Limit samples for performance
        if len(X_processed) > max_samples:
            indices = np.random.choice(len(X_processed), max_samples, replace=False)
            X_explain = X_processed[indices]
        else:
            X_explain = X_processed
            indices = np.arange(len(X_processed))
        
        # Generate SHAP values
        shap_values = explainer.shap_values(X_explain)
        
        # Handle multiclass output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        
        # Get predictions
        predictions = self.predict(X_explain)
        
        # Calculate feature importance
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        return {
            'shap_values': shap_values,
            'predictions': predictions,
            'explained_indices': indices,
            'feature_importance': feature_importance,
            'base_value': explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
            'model_type': 'randomforest'
        }


class IsolationForestExplainer(MLModelWrapper):
    """
    Isolation Forest wrapper with SHAP explanations for anomaly detection.
    
    Provides explanations for unsupervised anomaly detection using
    Isolation Forest with KernelExplainer for interpretability.
    """
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        preprocessing: bool = True
    ):
        """
        Initialize Isolation Forest explainer.
        
        Args:
            model_params: Isolation Forest specific parameters
            feature_names: Names of input features
            preprocessing: Whether to apply preprocessing
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with: pip install scikit-learn")
        
        default_params = {
            'contamination': 0.1,
            'n_estimators': 100,
            'max_features': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params, feature_names, preprocessing)
    
    def _create_model(self) -> IsolationForest:
        """Create Isolation Forest model."""
        return IsolationForest(**self.model_params)
    
    def _get_model_type(self) -> str:
        """Get model type identifier."""
        return "isolationforest"
    
    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train Isolation Forest model (unsupervised).
        
        Args:
            X_train: Training features
            y_train: Training labels (used only for evaluation)
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics
        """
        from datetime import datetime
        start_time = datetime.now()
        
        # Train model (unsupervised)
        self.model.fit(X_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Get anomaly predictions (-1 for anomaly, 1 for normal)
        y_train_pred = self.model.predict(X_train)
        # Convert to binary (0 for normal, 1 for anomaly)
        y_train_binary = (y_train_pred == -1).astype(int)
        
        # Calculate accuracy if labels provided
        train_accuracy = None
        if y_train is not None:
            train_accuracy = np.mean(y_train_binary == y_train)
        
        metrics = {
            'model_type': 'isolationforest',
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'contamination': self.model.contamination,
            'n_estimators': self.model.n_estimators,
            'anomaly_rate': np.mean(y_train_binary)
        }
        
        # Validation metrics if available
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            y_val_binary = (y_val_pred == -1).astype(int)
            
            metrics.update({
                'val_accuracy': np.mean(y_val_binary == y_val),
                'val_anomaly_rate': np.mean(y_val_binary)
            })
        
        return metrics
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using Isolation Forest model."""
        predictions = self.model.predict(X)
        # Convert to binary (0 for normal, 1 for anomaly)
        return (predictions == -1).astype(int)
    
    def _predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get anomaly scores as probabilities."""
        # Get decision function scores (higher means more normal)
        scores = self.model.decision_function(X)
        
        # Convert to probabilities (higher score = lower anomaly probability)
        # Normalize scores to [0, 1] range where 1 means anomaly
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            # All scores are the same
            probabilities = np.full((len(scores), 2), 0.5)
        else:
            # Normalize and invert (lower decision function score = higher anomaly probability)
            normalized_scores = (scores - min_score) / (max_score - min_score)
            anomaly_proba = 1 - normalized_scores
            normal_proba = normalized_scores
            
            probabilities = np.column_stack([normal_proba, anomaly_proba])
        
        return probabilities
    
    def create_shap_explainer(self, background_data: np.ndarray) -> 'shap.KernelExplainer':
        """
        Create SHAP KernelExplainer for Isolation Forest.
        
        Args:
            background_data: Background data for KernelExplainer
            
        Returns:
            SHAP KernelExplainer instance
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before creating explainer")
        
        # Preprocess background data if needed
        if self.preprocessing:
            background_processed = self._preprocess_features(background_data)
        else:
            background_processed = background_data
        
        # Create prediction function for SHAP
        def predict_fn(X):
            probabilities = self._predict_proba(X)
            return probabilities
        
        # KernelExplainer for Isolation Forest
        explainer = shap.KernelExplainer(predict_fn, background_processed)
        logger.info("Created SHAP KernelExplainer for Isolation Forest model")
        
        return explainer
    
    def explain_predictions(
        self,
        X: np.ndarray,
        background_data: np.ndarray,
        explainer: Optional['shap.KernelExplainer'] = None,
        max_samples: int = 50  # Lower for KernelExplainer performance
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for Isolation Forest predictions.
        
        Args:
            X: Input features to explain
            background_data: Background data for explainer
            explainer: Pre-created SHAP explainer
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP explanation results
        """
        if explainer is None:
            explainer = self.create_shap_explainer(background_data)
        
        # Preprocess input if needed
        X_processed = self._preprocess_features(X) if self.preprocessing else X
        
        # Limit samples for performance (KernelExplainer is slower)
        if len(X_processed) > max_samples:
            indices = np.random.choice(len(X_processed), max_samples, replace=False)
            X_explain = X_processed[indices]
        else:
            X_explain = X_processed
            indices = np.arange(len(X_processed))
        
        # Generate SHAP values (slower for KernelExplainer)
        logger.info(f"Generating SHAP explanations for {len(X_explain)} samples (this may take a while)...")
        shap_values = explainer.shap_values(X_explain)
        
        # Use anomaly class (index 1) for binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get predictions
        predictions = self.predict(X_explain)
        
        # Calculate feature importance
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        return {
            'shap_values': shap_values,
            'predictions': predictions,
            'explained_indices': indices,
            'feature_importance': feature_importance,
            'base_value': explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
            'model_type': 'isolationforest'
        }
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get raw anomaly scores from Isolation Forest.
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores (lower means more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting anomaly scores")
        
        X_processed = self._preprocess_features(X) if self.preprocessing else X
        return self.model.decision_function(X_processed)
