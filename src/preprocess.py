import pandas as pd

df = pd.read_csv("data/titanic.csv")
features = ["Pclass", "Sex", "SibSp", "Parch", "Survived"]
# Realizar algún preprocesamiento, por ejemplo, agregar una columna 'target'
df = df[features]
df_proces = pd.get_dummies(df)

df_proces.to_csv("data/processed_data.csv", index=False)
