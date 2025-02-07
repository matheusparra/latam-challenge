import fastapi
import pandas as pd
from pydantic import BaseModel
from typing import List
from model import DelayModel  # Importa o modelo de `model.py`

app = fastapi.FastAPI()

# Criando uma instância do modelo
model = DelayModel()

# Estrutura esperada para a entrada de um voo
class FlightData(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str
    SIGLADES: str
    DIANOM: str

@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Endpoint para checar se a API está rodando corretamente.
    """
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(flights: List[FlightData]) -> dict:
    """
    Endpoint para prever se um voo terá atraso ou não.
    """
    df = pd.DataFrame([item.dict() for item in flights])

    try:
        features = model.preprocess(df)  # Chama o método preprocess() do modelo
        predictions = model.predict(features)  # Chama o método predict()
        return {"predictions": predictions}
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))

@app.post("/train", status_code=200)
async def train_model(flights: List[FlightData], targets: List[int]):
    """
    Endpoint para treinar o modelo com novos dados.
    """
    df = pd.DataFrame([item.dict() for item in flights])
    df['delay'] = targets  # Adiciona a variável target

    features, target = model.preprocess(df, target_column="delay")
    model.fit(features, target)  # Chama o método fit()

    return {"message": "Modelo treinado com sucesso!"}
 