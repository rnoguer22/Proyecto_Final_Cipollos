import pandas as pd

#Preunta 1
df = pd.read_excel("Files/regression_data.xlsx")
print(df)

#Preginta2
df.drop(columns="date", inplace=True)