import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression

from typing import Tuple, Union, List


def get_min_diff(data):
    fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return min_diff


class DelayModel:
    def __init__(self):
        self._model = None  # Model should be saved in this attribute.
        self.threshold_in_minutes = 15  # delay threshold in minutes
        self.top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
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
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )
        missing_features = list(set(self.top_10_features) - set(features.columns))
        ## filling missing features with 0
        for mf in missing_features:
            features[mf] = 0

        preprocessed_data = features[self.top_10_features]

        if target_column is None:
            return preprocessed_data

        # calculate target
        data["min_diff"] = data.apply(get_min_diff, axis=1)
        data[target_column] = np.where(
            data["min_diff"] > self.threshold_in_minutes, 1, 0
        )

        return (preprocessed_data, data[[target_column]])

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        n_y0 = len(target[target["delay"] == 0])
        n_y1 = len(target[target["delay"] == 1])

        x = features[self.top_10_features]
        y = target["delay"]

        reg_model = LogisticRegression(
            class_weight={1: n_y0 / len(y), 0: n_y1 / len(y)}
        )
        reg_model.fit(x, y)
        self._model = reg_model
        print(reg_model)
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        x = features[self.top_10_features]
        y_pred = self._model.predict(x).tolist()
        return y_pred