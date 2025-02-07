import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Union, List

class DelayModel:
    def __init__(self):
        self._model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._encoder = LabelEncoder()
    
    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Preprocess the raw data.
        
        Args:
            data (pd.DataFrame): Raw input data.
            target_column (str, optional): Column to be used as target.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and target if target_column is provided.
            or
            pd.DataFrame: Processed features only.
        """
        data = data.copy()
        
        # Create 'high_season' column
        data['Date-I'] = pd.to_datetime(data['DIA'])
        data['high_season'] = data['DIA'].apply(lambda x: 1 if (x.month == 12 and x.day >= 15) or 
                                                               (x.month == 1 or x.month == 2) or 
                                                               (x.month == 3 and x.day <= 3) or 
                                                               (x.month == 7 and 15 <= x.day <= 31) or 
                                                               (x.month == 9 and 11 <= x.day <= 30) else 0)
        
        # Create 'min_diff' column
        data['Date-O'] = pd.to_datetime(data['Date-O'])
        data['min_diff'] = (data['Date-O'] - data['Date-I']).dt.total_seconds() / 60
        
        # Create 'period_day' column
        data['hour'] = data['Date-I'].dt.hour
        data['period_day'] = data['hour'].apply(lambda x: 'morning' if 5 <= x < 12 else 
                                                            'afternoon' if 12 <= x < 19 else 'night')
        data.drop(columns=['hour'], inplace=True)
        
        # Create 'delay' column
        data['delay'] = (data['min_diff'] > 15).astype(int)
        
        # Encode categorical features
        if 'OPERA' in data.columns:
            data['OPERA'] = self._encoder.fit_transform(data['OPERA'])
        if 'TIPOVUELO' in data.columns:
            data['TIPOVUELO'] = self._encoder.fit_transform(data['TIPOVUELO'])
        if 'period_day' in data.columns:
            data['period_day'] = self._encoder.fit_transform(data['period_day'])
        
        # Select features
        features = data[['OPERA', 'TIPOVUELO', 'MES', 'DIA', 'high_season', 'min_diff', 'period_day']]
        
        if target_column:
            target = data[target_column]
            return features, target
        return features
    
    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Train the model with given features and target.
        
        Args:
            features (pd.DataFrame): Processed feature set.
            target (pd.DataFrame): Target labels.
        """
        self._model.fit(features, target)
    
    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict flight delays.
        
        Args:
            features (pd.DataFrame): Processed feature set.
        
        Returns:
            List[int]: Predicted target labels.
        """
        return self._model.predict(features).tolist()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model.
        """
        joblib.dump((self._model, self._encoder), filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a pre-trained model.
        
        Args:
            filepath (str): Path to load the model.
        """
        self._model, self._encoder = joblib.load(filepath)

# Example usage
if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("C:/Users/parra/Documents/LATAM/data/data.csv")
    
    # Initialize model
    model = DelayModel()
    
    # Preprocess data
    features, target = model.preprocess(data, target_column='delay')
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save trained model
    model.save_model("delay_model.pkl")
