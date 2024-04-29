import pandas as pd

dataset = pd.read_csv("Sampel.csv", sep=';', quotechar='"')
print(dataset.head())

print(dataset['Kenyamanan'].unique())
