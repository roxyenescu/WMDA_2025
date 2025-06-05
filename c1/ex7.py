# [Scaling numerical features in a housing price dataset.]
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Step 1: Crearea unui set de date cu 5 case (4 coloane)
data = {
    "HouseID": [1, 2, 3, 4, 5],
    "SquareFeet": [1500, 1800, 1200, 2000, 1600],
    "NumRooms": [3, 4, 2, 5, 3],
    "Price": [250000, 300000, 200000, 350000, 280000]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# Step 2: Initializarea scalatoarelor
minmax_scaler = MinMaxScaler() # aduce fiecare coloana intre valorile 0 si 1
standard_scaler = StandardScaler() # centreaza valorile in jurul 0, cu deviatie standard 1

# Step 3: Aplicarea MinMax Scaling (scales values between 0 and 1)
# (x - x min) / (x max - x min)
df_minmax_scaled = df.copy() # copiez tabelul original intr-un nou tabel ca sa nu pierd datele initiale
df_minmax_scaled[["SquareFeet", "NumRooms", "Price"]] = minmax_scaler.fit_transform(df[["SquareFeet", "NumRooms", "Price"]])

print("\nDataset After Min-Max Scaling:\n")
print(df_minmax_scaled)

# Step 4: Aplicarea Standard Scaling (scales values to have mean = 0 and std deviation = 1)
# (x - media_coloanei) / deviatia std a coloanei
df_standard_scaled = df.copy()
df_standard_scaled[["SquareFeet", "NumRooms", "Price"]] = standard_scaler.fit_transform(df[["SquareFeet", "NumRooms", "Price"]])

print("\nDataset After Standard Scaling:\n")
print(df_standard_scaled) # afisarea valorilor standardizate
