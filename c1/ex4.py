# [Querying a Sqlite database to retrieve user transaction data.]
import sqlite3

# Step 1: Conectarea la o abza de date SQLite
conn = sqlite3.connect("transactions.db") # conectarea la o baza de date sqlite
cursor = conn.cursor() # initializarea cursorului

# Step 2: Creeaza o noua tabela cu coloane in cazul in care aceasta nu exista
cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        amount REAL,
        transaction_type TEXT,
        date TEXT
    )
''')

# Step 3: Adauga date de test daca tabela este goala
cursor.execute("SELECT COUNT(*) FROM transactions")
if cursor.fetchone()[0] == 0: # verifica daca tabela este goala (se obtine nr de randuri)
    sample_data = [
        (1, 250.75, 'deposit', '2024-02-23'),
        (1, -100.50, 'withdrawal', '2024-02-24'),
        (2, 500.00, 'deposit', '2024-02-20'),
        (3, -75.00, 'withdrawal', '2024-02-21'),
    ] # se creeaza o lista cu tranzactii pt 3 utilizatori
    cursor.executemany(""
                       "INSERT INTO transactions (user_id, amount, transaction_type, date) VALUES (?, ?, ?, ?)",
                       sample_data
                       ) # se executa comanda pt mai multe randuri
    conn.commit() # salveaza schimbarile in baza de date

# Step 4: Selecteaza tranzactiile unui utilizator (e.g., transactions for user_id = 1)
user_id = 1 # vrem sa vedem tranzactiile utilizatorului 1
cursor.execute(
    "SELECT * FROM transactions WHERE user_id = ?",
    (user_id,)
) # obtinem toate randurile unde id-ul utilizatorului este 1
transactions = cursor.fetchall() # se intoarce o lista cu toate rezultatele

# Step 5: Afisarea rezultatelor (Afisarea fiecarei tranzactii)
print(f"Transactions for User {user_id}:\n")
for txn in transactions:
    print(f"ID: {txn[0]}, Amount: {txn[2]}, Type: {txn[3]}, Date: {txn[4]}")

# Inchiderea conexiunii la baza de date
conn.close()
