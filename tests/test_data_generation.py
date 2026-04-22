"""Tests for data generation and processing."""

import pytest
import numpy as np
import pandas as pd

from dp_implementation.data.generator import DataGenerator
from dp_implementation.data.processor import DataProcessor


class TestDataGenerator:
    """Test class for data generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_generator = DataGenerator(random_seed=42)
    
    def test_generate_synthetic_classification(self):
        """Test synthetic classification data generation."""
        n_samples = 100
        n_features = 5
        n_classes = 2
        
        X, y = self.data_generator.generate_synthetic_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes
        )
        
        # Check shapes
        assert X.shape == (n_samples, n_features)
        assert y.shape == (n_samples,)
        
        # Check data types
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        
        # Check class distribution
        unique_classes = np.unique(y)
        assert len(unique_classes) == n_classes
        assert all(cls in range(n_classes) for cls in unique_classes)
    
    def test_generate_synthetic_regression(self):
        """Test synthetic regression data generation."""
        n_samples = 100
        n_features = 5
        
        X, y = self.data_generator.generate_synthetic_regression(
            n_samples=n_samples,
            n_features=n_features
        )
        
        # Check shapes
        assert X.shape == (n_samples, n_features)
        assert y.shape == (n_samples,)
        
        # Check data types
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        
        # Check that y is continuous
        assert len(np.unique(y)) > 10  # Should have many unique values
    
    def test_generate_income_data(self):
        """Test income data generation."""
        n_samples = 100
        
        income_data = self.data_generator.generate_income_data(n_samples=n_samples)
        
        # Check shape
        assert len(income_data) == n_samples
        
        # Check data type
        assert isinstance(income_data, pd.DataFrame)
        
        # Check required columns
        required_columns = ["age", "education", "experience_years", "income", "zip_code", "ssn_last4"]
        for col in required_columns:
            assert col in income_data.columns
        
        # Check data ranges
        assert income_data["age"].min() >= 18
        assert income_data["age"].max() <= 65
        assert income_data["income"].min() >= 20000  # Minimum wage
    
    def test_generate_health_data(self):
        """Test health data generation."""
        n_samples = 100
        
        health_data = self.data_generator.generate_health_data(n_samples=n_samples)
        
        # Check shape
        assert len(health_data) == n_samples
        
        # Check data type
        assert isinstance(health_data, pd.DataFrame)
        
        # Check required columns
        required_columns = ["patient_id", "age", "gender", "bmi", "blood_pressure_systolic", 
                           "blood_pressure_diastolic", "cholesterol", "diabetes", "heart_disease"]
        for col in required_columns:
            assert col in health_data.columns
        
        # Check data ranges
        assert health_data["age"].min() >= 18
        assert health_data["age"].max() <= 80
        assert health_data["gender"].isin(["M", "F"]).all()
    
    def test_generate_location_data(self):
        """Test location data generation."""
        n_samples = 100
        
        location_data = self.data_generator.generate_location_data(n_samples=n_samples)
        
        # Check shape
        assert len(location_data) == n_samples
        
        # Check data type
        assert isinstance(location_data, pd.DataFrame)
        
        # Check required columns
        required_columns = ["user_id", "timestamp", "latitude", "longitude"]
        for col in required_columns:
            assert col in location_data.columns
        
        # Check data ranges
        assert location_data["latitude"].min() >= 25.0
        assert location_data["latitude"].max() <= 49.0
        assert location_data["longitude"].min() >= -125.0
        assert location_data["longitude"].max() <= -66.0
    
    def test_generate_transaction_data(self):
        """Test transaction data generation."""
        n_samples = 100
        
        transaction_data = self.data_generator.generate_transaction_data(n_samples=n_samples)
        
        # Check shape
        assert len(transaction_data) == n_samples
        
        # Check data type
        assert isinstance(transaction_data, pd.DataFrame)
        
        # Check required columns
        required_columns = ["user_id", "merchant_id", "timestamp", "amount", "category"]
        for col in required_columns:
            assert col in transaction_data.columns
        
        # Check data ranges
        assert transaction_data["amount"].min() >= 1.0
        assert transaction_data["amount"].max() <= 1000.0
    
    def test_create_train_test_split(self):
        """Test train-test split creation."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = self.data_generator.create_train_test_split(
            X, y, test_size=0.2
        )
        
        # Check shapes
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20
        
        # Check that features have same number of columns
        assert X_train.shape[1] == X_test.shape[1] == 5
    
    def test_create_time_series_split(self):
        """Test time-based split creation."""
        # Create sample time series data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(100)
        })
        
        train_data, val_data, test_data = self.data_generator.create_time_series_split(
            data, 'timestamp', train_ratio=0.7, val_ratio=0.15
        )
        
        # Check shapes
        assert len(train_data) == 70
        assert len(val_data) == 15
        assert len(test_data) == 15
        
        # Check that data is sorted by time
        assert train_data['timestamp'].max() <= val_data['timestamp'].min()
        assert val_data['timestamp'].max() <= test_data['timestamp'].min()


class TestDataProcessor:
    """Test class for data processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_processor = DataProcessor(random_seed=42)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
            'zip_code': [12345, 23456, 34567, 45678, 56789],
            'ssn_last4': [1234, 2345, 3456, 4567, 5678]
        })
    
    def test_anonymize_identifiers(self):
        """Test identifier anonymization."""
        id_columns = ['zip_code', 'ssn_last4']
        
        anonymized_data = self.data_processor.anonymize_identifiers(
            self.sample_data, id_columns, method="hash"
        )
        
        # Check that original columns are still present
        for col in id_columns:
            assert col in anonymized_data.columns
        
        # Check that values are different (anonymized)
        for col in id_columns:
            assert not anonymized_data[col].equals(self.sample_data[col])
    
    def test_remove_pii(self):
        """Test PII removal."""
        pii_columns = ['ssn_last4']
        
        cleaned_data = self.data_processor.remove_pii(self.sample_data, pii_columns)
        
        # Check that PII columns are removed
        for col in pii_columns:
            assert col not in cleaned_data.columns
        
        # Check that other columns are preserved
        assert 'age' in cleaned_data.columns
        assert 'income' in cleaned_data.columns
        assert 'education' in cleaned_data.columns
    
    def test_hash_sensitive_fields(self):
        """Test sensitive field hashing."""
        sensitive_columns = ['ssn_last4']
        
        hashed_data = self.data_processor.hash_sensitive_fields(
            self.sample_data, sensitive_columns
        )
        
        # Check that sensitive columns are hashed
        for col in sensitive_columns:
            assert col in hashed_data.columns
            # Check that values are different (hashed)
            assert not hashed_data[col].equals(self.sample_data[col])
    
    def test_normalize_features(self):
        """Test feature normalization."""
        numeric_columns = ['age', 'income']
        
        normalized_data = self.data_processor.normalize_features(
            self.sample_data, method="standard", columns=numeric_columns
        )
        
        # Check that normalized columns have mean ~0 and std ~1
        for col in numeric_columns:
            assert abs(normalized_data[col].mean()) < 0.1
            assert abs(normalized_data[col].std() - 1.0) < 0.1
    
    def test_encode_categorical(self):
        """Test categorical encoding."""
        categorical_columns = ['education']
        
        encoded_data = self.data_processor.encode_categorical(
            self.sample_data, columns=categorical_columns, method="label"
        )
        
        # Check that categorical columns are encoded
        for col in categorical_columns:
            assert col in encoded_data.columns
            # Check that values are numeric
            assert encoded_data[col].dtype in ['int64', 'int32']
    
    def test_add_noise(self):
        """Test noise addition."""
        numeric_columns = ['age', 'income']
        noise_level = 0.1
        
        noisy_data = self.data_processor.add_noise(
            self.sample_data, noise_level=noise_level, noise_type="gaussian"
        )
        
        # Check that values are different (noise added)
        for col in numeric_columns:
            assert not noisy_data[col].equals(self.sample_data[col])
    
    def test_create_entity_aware_split(self):
        """Test entity-aware split creation."""
        # Add entity column
        data_with_entity = self.sample_data.copy()
        data_with_entity['entity_id'] = [1, 1, 2, 2, 3]
        
        train_data, val_data, test_data = self.data_processor.create_entity_aware_split(
            data_with_entity, 'entity_id', test_ratio=0.2, val_ratio=0.2
        )
        
        # Check that entities are not split across sets
        train_entities = set(train_data['entity_id'])
        val_entities = set(val_data['entity_id'])
        test_entities = set(test_data['entity_id'])
        
        assert len(train_entities.intersection(val_entities)) == 0
        assert len(train_entities.intersection(test_entities)) == 0
        assert len(val_entities.intersection(test_entities)) == 0
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        summary = self.data_processor.get_data_summary(self.sample_data)
        
        # Check that summary contains required keys
        required_keys = ['shape', 'columns', 'dtypes', 'missing_values', 'numeric_summary', 'categorical_summary']
        for key in required_keys:
            assert key in summary
        
        # Check shape
        assert summary['shape'] == self.sample_data.shape
        
        # Check columns
        assert summary['columns'] == self.sample_data.columns.tolist()
