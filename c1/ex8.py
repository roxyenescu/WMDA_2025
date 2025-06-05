# [Encoding gender (male/female) as binary values.]
import pandas as pd # lucrul cu tabele

# Step 1: Crearea unui set de date cu o coloana pentru "Gender"
data = {
    "PersonID": [1, 2, 3, 4, 5],
    "Name": ["John", "Emma", "Liam", "Sophia", "Noah"],
    "Gender": ["Male", "Female", "Male", "Female", "Male"]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# Step 2: Convertirea valorilor de tip text din coloana "Gender" in valori binare (Male=0, Female=1)
df["Gender_Binary"] = df["Gender"].map({"Male": 0, "Female": 1})

print("\nDataset After Encoding Gender as Binary:\n")
print(df)
