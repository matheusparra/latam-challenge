from challenge.model import DelayModel
import pandas as pd

# Criando um conjunto de dados de teste
data = pd.DataFrame({
    'OPERA': ['LATAM'],
    'MES': [7],
    'TIPOVUELO': ['I'],
    'SIGLADES': ['Santiago'],
    'DIANOM': ['Thursday'],
    'delay': [1]
},{'OPERA': ['LATAM'],
    'MES': [7],
    'TIPOVUELO': ['I'],
    'SIGLADES': ['Santiago'],
    'DIANOM': ['Thursday'],
    'delay': [1]})
print("Colunas disponíveis no DataFrame:", data.columns)
# Criando o modelo
model = DelayModel()

# Preprocessar e treinar o modelo
features, target = model.preprocess(data, target_column="delay")
model.fit(features, target)

# Criando novos dados para prever
novos_dados = model.preprocess(data.drop(columns=["delay"]))
previsoes = model.predict(novos_dados)

print("Previsões:", previsoes)