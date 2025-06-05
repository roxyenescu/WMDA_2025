# [Handling missing age values in a Titanic survival dataset.]
import pandas as pd # folosit pentru lucrul cu tabele de date
import numpy as np # folosit pt a adauga valori lipsa

# Step 1: Crearea unui set de date cu pasageri de pe Titanic cu valori de varsta lipsa
data = {
    "PassengerID": [1, 2, 3, 4, 5],
    "Name": ["John", "Emma", "Liam", "Sophia", "Noah"],
    "Age": [22, np.nan, 24, np.nan, 30],  # valori lipsa in coloana de Varsta
    "Survived": [1, 0, 1, 1, 0] # 1 - daca a supravietuit, 0 - daca nu
}

df = pd.DataFrame(data) # se creeaza un tabel pe baza dictionarului

print("Original Dataset with Missing Age Values:\n")
print(df) # se afiseaza tabelul original cu valorile lipsa

# Step 2: Inlocuieste valorile lipsa cu media coloanei
df["Age_mean"] = df["Age"].fillna(df["Age"].mean())

# Step 3: Inlocuieste valorile lipsa cu mediana coloanei
df["Age_median"] = df["Age"].fillna(df["Age"].median())

# Step 4: Inlocuieste valorile lipsa cu modulul coloanei (cea mai frecventa valoare)
df["Age_mode"] = df["Age"].fillna(df["Age"].mode()[0]) # [0] - ia primul mod daca sunt mai multi cu aceeasi frecventa

print("\nDataset After Handling Missing Age Values (Mean, Median, Mode):\n")
print(df[["PassengerID", "Name", "Age", "Age_mean", "Age_median", "Age_mode"]])
