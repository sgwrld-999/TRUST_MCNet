
"""
data_ploader.py

Provides configuration management, JSON-based logging, data preprocessing,
splitting and PyTorch DataLoader construction for a deep learning workflow.
"""

import os, yaml, json, logging
from pathlib import Path
from datetime import datetime
from typing import Any, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

class ConfigManager:
    """
    Handles loading and retrieval of hierarchical configuration values from YAML.

    Reads a YAML file at initialization ('config.yaml') and provides
    nested key lookup with dot notation.

    Attributes:
        cfg (dict): Loaded configuration dictionary.
    """
    def __init__(self, path="config.yaml"):
        """
        Args:
            path: Filesystem path to the YAML configuration file.
        """
        p = Path(path)
        self.cfg = yaml.safe_load(p.read_text()) if p.exists() else {}
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value using dot-separated keys.

        Args:
            key: Dot-notation path (e.g. "data.target_column").
            default: Value to return if key is missing.

        Returns:
            The configuration value, or default if not found.
        """
        v = self.cfg
        for k in key.split("."):
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                return default
        return v

class JsonLogger:
    """
    A simple JSON-format logger that writes timestamped entries to a file.

    Attributes:
        l (logging.Logger): Underlying Python logger instance.
    """
    def __init__(self, file: str, level: str = "INFO"):
        """
        Args:
            file: Path to the log file. Parent directories will be created.
            level: Logging level as a string (e.g. "DEBUG", "INFO").
        """
        p = Path(file); p.parent.mkdir(parents=True, exist_ok=True)
        self.l = logging.getLogger("dl"); self.l.setLevel(getattr(logging, level))
        h = logging.FileHandler(file, "w"); h.setFormatter(logging.Formatter("%(message)s"))
        self.l.handlers = [h]
    def log(self, level: str, msg: str, **kw):
        """
        Write a JSON-encoded log entry with timestamp, level and message.

        Args:
            level: Severity level (e.g. "INFO", "ERROR").
            msg: Human-readable message.
            **kw: Additional key-value pairs to include in the entry.
        """
        entry = {"timestamp": datetime.now().isoformat(), "level": level, "message": msg, **kw}
        getattr(self.l, level.lower())(json.dumps(entry))

class DataPreprocessor:
    """
    Encapsulates feature encoding, imputation, scaling and label encoding.

    On fit, determines numeric and categorical columns, fits transformers
    (SimpleImputer + StandardScaler) for numeric data and LabelEncoder
    for each categorical feature.

    Attributes:
        cm: ConfigManager instance for retrieving settings.
        lg: JsonLogger instance for structured logging.
        le (dict[str, LabelEncoder]): Fitted encoders for categorical columns.
        ct (ColumnTransformer): Combined numeric transformer pipeline.
    """
    def __init__(self, cm: ConfigManager, lg: JsonLogger):
        """
        Args:
            cm: ConfigManager instance.
            lg: JsonLogger instance.
        """
        self.cm, self.lg, self.le, self.ct = cm, lg, {}, None
    def fit(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit transformers on the DataFrame and transform features.

        Args:
            df: Pandas DataFrame containing features and target.

        Returns:
            A tuple (X_transformed, y_array), where X_transformed is a numpy
            array of transformed features, and y_array is the target values.
        """
        tc = self.cm.get("data.target_column")
        y = df[tc].values if tc and tc in df else None
        X = df.drop(columns=[tc]) if tc and tc in df else df.copy()
        num = [c for c in X if pd.api.types.is_numeric_dtype(X[c])]
        cat = [c for c in X if c not in num]
        for c in cat:
            le = LabelEncoder().fit(X[c].astype(str)); self.le[c] = le
            X[c] = le.transform(X[c].astype(str))
        if num:
            self.ct = ColumnTransformer([
                ("n", Pipeline([("im", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), num)
            ], remainder="drop")
            Xt = self.ct.fit_transform(X)
        else:
            Xt = X.values
        return Xt, y
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply fitted transformations to new data.

        Args:
            df: New DataFrame with same schema as used in fit().

        Returns:
            Numpy array of transformed features.
        """
        tc = self.cm.get("data.target_column")
        X = df.drop(columns=[tc]) if tc and tc in df else df.copy()
        for c, le in self.le.items():
            X[c] = le.transform(X[c].astype(str))
        return self.ct.transform(X) if self.ct else X.values

class ClientDataset(Dataset):
    """
    PyTorch Dataset wrapping feature and label arrays.

    Converts numpy arrays to torch tensors of appropriate dtype.

    Attributes:
        X (torch.Tensor): Features of shape (N, D), dtype float32.
        y (torch.Tensor): Integer class labels of shape (N,), dtype long.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: 2D numpy array of input features.
            y: 1D numpy array of target labels.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): 
        """Return the number of samples."""
        return len(self.X)
    def __getitem__(self, i): 
        """
        Retrieve one sample by index.

        Args:
            idx: Index of the sample.

        Returns:
            A tuple (feature_tensor, label_tensor).
        """
        return self.X[i], self.y[i]

class DataSplitter:
    """
    Splits arrays into train, test and validation sets according to configuration.

    Uses sklearn.model_selection.train_test_split with optional stratification.

    Attributes:
        cm: ConfigManager instance for retrieving split ratios.
        lg: JsonLogger instance for recording split sizes.
    """
    def __init__(self, cm: ConfigManager, lg: JsonLogger):
        """
        Args:
            cm: Configuration manager.
            lg: Logger for recording the split operation.
        """
        self.cm, self.lg = cm, lg
    def split(self, X: np.ndarray, y: np.ndarray):
        """
        Perform train/test/validation split.

        Reads ratios 'split.train_ratio', 'split.test_ratio' and 'split.val_ratio',
        ensures they sum to 1, and applies two-stage splitting. Logs counts.

        Args:
            X: Feature array.
            y: Label array.

        Returns:
            A tuple of three (X_sub, y_sub) pairs:
            (train, test, validation).
        """
        tr, te, va = (self.cm.get(k, d) for k, d in [
            ("split.train_ratio", 0.8),
            ("split.test_ratio", 0.15),
            ("split.val_ratio", 0.05),
        ])
        rs = self.cm.get("split.random_state", 42) # random_state
        st = self.cm.get("split.stratify", True) and y is not None # stratify_flag
        assert abs(tr + te + va - 1) < 1e-6 # Ensure ratios sum to 1
        tv = te + va

        # First split: train vs. temp (test+val)
        xtr, xtv, ytr, ytv = train_test_split(X, y, test_size=tv,
                                              stratify=y if st else None, random_state=rs)
        v = va / tv
        # Second split: test vs. val
        xte, xva, yte, yva = train_test_split(xtv, ytv, test_size=v,
                                              stratify=ytv if st else None, random_state=rs)
        self.lg.log("INFO", "split", train=len(xtr), test=len(xte), val=len(xva))
        return (xtr, ytr), (xte, yte), (xva, yva)

class DataLoaderFactory:
    """
    Orchestrates end-to-end data loading: reading CSV, preprocessing, splitting,
    and wrapping in PyTorch DataLoaders.

    Configuration options:
        data.file_path, data.target_column,
        logging.log_file, logging.log_level,
        split.* ratios and random_state.

    Attributes:
        cm: ConfigManager instance.
        lg: JsonLogger instance.
        dp: DataPreprocessor instance.
        sp: DataSplitter instance.
    """
    def __init__(self, cfg: str = "config.yaml"):
        """
        Args:
            cfg: Path to configuration YAML.
        """
        self.cm = ConfigManager(cfg)
        self.lg = JsonLogger(self.cm.get("logging.log_file", "logs/data_loader.json"),
                             self.cm.get("logging.log_level", "INFO"))
        self.dp = DataPreprocessor(self.cm, self.lg)
        self.sp = DataSplitter(self.cm, self.lg)
        self.lg.log("INFO", "init")
    def load(self) -> pd.DataFrame:
        """
        Read the raw data CSV into a DataFrame.

        Returns:
            Pandas DataFrame of the entire dataset.
        """
        fp = self.cm.get("data.file_path", "data/clients.csv")
        df = pd.read_csv(fp)
        self.lg.log("INFO", "loaded", shape=df.shape)
        return df
    def create(self, bs: int = 32, nw: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build DataLoaders for training, validation and testing.

        Args:
            bs: Batch size for all DataLoaders.
            nw: Number of worker processes for loading.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        df = self.load()
        tc = self.cm.get("data.target_column")
        df = df.dropna(subset=[tc]) if tc and tc in df else df
        Xt, y = self.dp.fit(df)
        (xtr, ytr), (xte, yte), (xva, yva) = self.sp.split(Xt, y)
        ds = lambda X, y: ClientDataset(X, y)
        loaders = [
            DataLoader(ds(xtr, ytr), batch_size=bs, shuffle=True, num_workers=nw),
            DataLoader(ds(xva, yva), batch_size=bs, shuffle=False, num_workers=nw),
            DataLoader(ds(xte, yte), batch_size=bs, shuffle=False, num_workers=nw),
        ]
        self.lg.log("INFO", "loaders", batch_size=bs, workers=nw)
        return tuple(loaders)

if __name__ == "__main__":
    """
    Example usage:
        $ python data_pipeline.py
    Prints the number of batches in each DataLoader.
    """
    tl, vl, el = DataLoaderFactory().create(64)
    print([len(l) for l in (tl, vl, el)])
