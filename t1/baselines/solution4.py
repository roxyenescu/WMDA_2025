### **Exercise 4: Feature Engineering for Classification**
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Incarcarea setului de date Titanic
df = sns.load_dataset("titanic")
print("Original Dataset Titanic: \n")
print(df)

# Crearea unui nou feature
df["family_size"] = df["sibsp"] + df["parch"] + 1

# Inlocuirea valorilor lipsa cu modulul coloanei
df["age"].fillna(df["age"].median())
df["embarked"].fillna(df["embarked"].mode()[0])

# Decodarea variabilelor de tip categorie (Sex and embarked)
df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

# Selectarea caracteristicilor numerice pentru scalare
scaler = MinMaxScaler()
df[["age", "fare", "family_size"]] = scaler.fit_transform(df[["age", "fare", "family_size"]])

# Afisarea setului de date curatat si transformat
print("Processed Titanic Dataset (First 5 Rows):\n")
print(df[["age", "fare", "family_size", "sex_male", "embarked_Q", "embarked_S"]].head())