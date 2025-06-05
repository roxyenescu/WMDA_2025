# [Cleaning a customer database by removing duplicate records.]
import pandas as pd # biblioteca folosita pentru a lucra cu datele in tabele (DataFrame-uri)

# Step 1: Crearea unei baze de date cu duplicate
data = {
    "customer_id": [101, 102, 103, 101, 104, 102],
    "name": ["Alice", "Bob", "Charlie", "Alice", "David", "Bob"],
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com",
              "alice@example.com", "david@example.com", "bob@example.com"]
}

df = pd.DataFrame(data) # convertesc dictionarul intr-un DataFrame (tabel de date)

print("Original Customer Database:\n")
print(df) # afisarea bazei de date (originala)

# Step 2: Eliminarea inregistrarilor duplicate care au coloane identice
df_cleaned = df.drop_duplicates() # elimina randurile in care toate valorile sunt identice, pastrand prima aparitie

print("\nDatabase After Removing Exact Duplicates:\n")
print(df_cleaned)

# Step 3: Elimina inregistrarile duplicate pe baza unei anumite coloane (ex: customer_id)
df_cleaned = df.drop_duplicates(subset=["customer_id"], keep="first")

print("\nDatabase After Removing Duplicates Based on Customer ID:\n")
print(df_cleaned)
