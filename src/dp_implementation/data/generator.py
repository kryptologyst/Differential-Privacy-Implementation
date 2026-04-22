"""Synthetic data generation for differential privacy experiments."""

import hashlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


class DataGenerator:
    """Generator for synthetic datasets for differential privacy experiments.
    
    This class creates various types of synthetic data that can be used to
    test and evaluate differential privacy mechanisms while avoiding real
    sensitive data.
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize data generator.
        
        Args:
            random_seed: Random seed for reproducible data generation.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_synthetic_classification(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 2,
        n_informative: int = 5,
        n_redundant: int = 2,
        noise: float = 0.1,
        class_sep: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification dataset.
        
        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features.
            n_classes: Number of classes.
            n_informative: Number of informative features.
            n_redundant: Number of redundant features.
            noise: Amount of noise in the data.
            class_sep: Class separation parameter.
            
        Returns:
            Tuple of (features, labels).
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            class_sep=class_sep,
            noise=noise,
            random_state=self.random_seed
        )
        
        return X, y
    
    def generate_synthetic_regression(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_informative: int = 5,
        noise: float = 0.1,
        bias: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic regression dataset.
        
        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features.
            n_informative: Number of informative features.
            noise: Amount of noise in the data.
            bias: Bias term.
            
        Returns:
            Tuple of (features, targets).
        """
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            bias=bias,
            random_state=self.random_seed
        )
        
        return X, y
    
    def generate_income_data(
        self,
        n_samples: int = 1000,
        age_range: Tuple[int, int] = (18, 65),
        education_levels: List[str] = None
    ) -> pd.DataFrame:
        """Generate synthetic income dataset.
        
        Args:
            n_samples: Number of samples to generate.
            age_range: Age range (min, max).
            education_levels: List of education levels.
            
        Returns:
            DataFrame with synthetic income data.
        """
        if education_levels is None:
            education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
        
        # Generate basic demographics
        ages = np.random.randint(age_range[0], age_range[1] + 1, n_samples)
        education = np.random.choice(education_levels, n_samples)
        experience = np.maximum(0, ages - 22 + np.random.normal(0, 2, n_samples))
        
        # Generate income based on demographics
        base_income = 30000
        age_factor = (ages - 25) * 1000
        education_factor = np.array([20000, 35000, 50000, 70000])[
            [education_levels.index(ed) for ed in education]
        ]
        experience_factor = experience * 1500
        
        income = base_income + age_factor + education_factor + experience_factor
        income += np.random.normal(0, 10000, n_samples)
        income = np.maximum(20000, income)  # Minimum wage
        
        # Add some privacy-sensitive attributes
        zip_codes = np.random.randint(10000, 99999, n_samples)
        ssn_last4 = np.random.randint(1000, 9999, n_samples)
        
        return pd.DataFrame({
            "age": ages,
            "education": education,
            "experience_years": experience,
            "income": income,
            "zip_code": zip_codes,
            "ssn_last4": ssn_last4
        })
    
    def generate_health_data(
        self,
        n_samples: int = 1000,
        age_range: Tuple[int, int] = (18, 80)
    ) -> pd.DataFrame:
        """Generate synthetic health dataset.
        
        Args:
            n_samples: Number of samples to generate.
            age_range: Age range (min, max).
            
        Returns:
            DataFrame with synthetic health data.
        """
        ages = np.random.randint(age_range[0], age_range[1] + 1, n_samples)
        genders = np.random.choice(["M", "F"], n_samples)
        
        # Generate health metrics
        bmi = np.random.normal(25, 5, n_samples)
        blood_pressure_systolic = np.random.normal(120, 15, n_samples)
        blood_pressure_diastolic = np.random.normal(80, 10, n_samples)
        cholesterol = np.random.normal(200, 40, n_samples)
        
        # Generate conditions based on age and other factors
        diabetes_prob = 0.1 + (ages - 50) * 0.01
        diabetes = np.random.binomial(1, np.clip(diabetes_prob, 0, 0.3), n_samples)
        
        heart_disease_prob = 0.05 + (ages - 60) * 0.015
        heart_disease = np.random.binomial(1, np.clip(heart_disease_prob, 0, 0.4), n_samples)
        
        # Generate patient IDs (hashed for privacy)
        patient_ids = [hashlib.sha256(str(i).encode()).hexdigest()[:8] for i in range(n_samples)]
        
        return pd.DataFrame({
            "patient_id": patient_ids,
            "age": ages,
            "gender": genders,
            "bmi": bmi,
            "blood_pressure_systolic": blood_pressure_systolic,
            "blood_pressure_diastolic": blood_pressure_diastolic,
            "cholesterol": cholesterol,
            "diabetes": diabetes,
            "heart_disease": heart_disease
        })
    
    def generate_location_data(
        self,
        n_samples: int = 1000,
        lat_range: Tuple[float, float] = (25.0, 49.0),
        lon_range: Tuple[float, float] = (-125.0, -66.0)
    ) -> pd.DataFrame:
        """Generate synthetic location dataset.
        
        Args:
            n_samples: Number of samples to generate.
            lat_range: Latitude range (min, max).
            lon_range: Longitude range (min, max).
            
        Returns:
            DataFrame with synthetic location data.
        """
        latitudes = np.random.uniform(lat_range[0], lat_range[1], n_samples)
        longitudes = np.random.uniform(lon_range[0], lon_range[1], n_samples)
        
        # Generate timestamps
        timestamps = pd.date_range(
            start="2023-01-01",
            end="2023-12-31",
            periods=n_samples
        )
        
        # Generate user IDs (hashed for privacy)
        user_ids = [hashlib.sha256(str(i).encode()).hexdigest()[:8] for i in range(n_samples)]
        
        return pd.DataFrame({
            "user_id": user_ids,
            "timestamp": timestamps,
            "latitude": latitudes,
            "longitude": longitudes
        })
    
    def generate_transaction_data(
        self,
        n_samples: int = 1000,
        amount_range: Tuple[float, float] = (1.0, 1000.0)
    ) -> pd.DataFrame:
        """Generate synthetic transaction dataset.
        
        Args:
            n_samples: Number of samples to generate.
            amount_range: Transaction amount range (min, max).
            
        Returns:
            DataFrame with synthetic transaction data.
        """
        amounts = np.random.uniform(amount_range[0], amount_range[1], n_samples)
        categories = np.random.choice(
            ["Food", "Transport", "Entertainment", "Shopping", "Bills", "Other"],
            n_samples
        )
        
        # Generate timestamps
        timestamps = pd.date_range(
            start="2023-01-01",
            end="2023-12-31",
            periods=n_samples
        )
        
        # Generate user and merchant IDs (hashed for privacy)
        user_ids = [hashlib.sha256(str(i).encode()).hexdigest()[:8] for i in range(n_samples)]
        merchant_ids = [hashlib.sha256(str(i + 10000).encode()).hexdigest()[:8] for i in range(n_samples)]
        
        return pd.DataFrame({
            "user_id": user_ids,
            "merchant_id": merchant_ids,
            "timestamp": timestamps,
            "amount": amounts,
            "category": categories
        })
    
    def create_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create train-test split.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of data for testing.
            random_state: Random state for reproducibility.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        if random_state is None:
            random_state = self.random_seed
        
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    def create_time_series_split(
        self,
        data: pd.DataFrame,
        time_col: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-based train-validation-test split.
        
        Args:
            data: DataFrame with time column.
            time_col: Name of time column.
            train_ratio: Proportion for training.
            val_ratio: Proportion for validation.
            
        Returns:
            Tuple of (train_data, val_data, test_data).
        """
        data_sorted = data.sort_values(time_col)
        n_samples = len(data_sorted)
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_data = data_sorted.iloc[:train_end]
        val_data = data_sorted.iloc[train_end:val_end]
        test_data = data_sorted.iloc[val_end:]
        
        return train_data, val_data, test_data
