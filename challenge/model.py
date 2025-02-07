import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List

warnings.filterwarnings('ignore')

class DelayModel:

    def __init__(
        self
    ):
        self._model = LogisticRegression 
        self.feature_columns = None

    @staticmethod
    def get_period_day(date):
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()
        
        if(date_time > morning_min and date_time < morning_max):
            return 'Moning'
        elif(date_time > afternoon_min and date_time < afternoon_max):
            return 'Afternoon'
        elif(
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
        ):
            return 'Night'
    @staticmethod    
    def is_high_season(fecha):
        fecha_ano = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_ano)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_ano)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_ano)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_ano)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_ano)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_ano)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_ano)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_ano)
    
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0
    @staticmethod
    def get_min_diff(data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff
    
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data['min_diff'] = data.apply(self.get_min_diff, axis = 1)
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        features = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM','delay']],random_state=111)
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )
        target = data['delay']

        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)
        print(f"train shape: {x_train.shape} | test shape: {x_test.shape}")
        y_train.value_counts('%')*100
        y_test.value_counts('%')*100
        # Se o modelo já foi treinado, reindexar para garantir colunas consistentes
        if self.feature_columns is not None:
            features = features.reindex(columns=self.feature_columns, fill_value=0)
        

        #target column
        if target_column:
            target = data[target_column]
            return features, target
        
        return features
    
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        self.feature_columns = list(features.columns)
        self._model = LogisticRegression(class_weight={1: len(y_train[y_train == 0]), 0: len(y_train[y_train == 1])})
        self._model.fit(x_train, y_train)
        
        y_pred = self._model.predict(x_test)
        print("Relatório de Classificação:")
        print(classification_report(y_test, y_pred))
        print("Matriz de Confusão:")
        print(confusion_matrix(y_test, y_pred))

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if not self._model:
            raise ValueError("The model has not been trained yet..")
        
        if self.feature_columns is None:
            raise ValueError("The model has not been trained yet..")
        
        if isinstance(features, tuple):  # Evita erro caso preprocess retorne um tuple
            features = features[0]

     
        features = features.reindex(columns=self.feature_columns,fill_value=0)
        print("resultado do features:", features.shape)
        predictions = self._model.predict(features)    
    
        return predictions.tolist()