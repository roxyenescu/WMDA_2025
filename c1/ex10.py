# [Dealing with class imbalance in fraud detection datasets.]
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE # crearea exemplelor fictive pt clasa minoritara
from imblearn.under_sampling import RandomUnderSampler # reduce nr de exemple pt clasa majoritara
from collections import Counter # numara cate exemple avem din fiecare clasa (0 - non-fraud, 1 - fraud)

# Step 1: Crearea unui set de date dezechilibrat
np.random.seed(42) # fixează aleatorietatea pentru a obține aceleași rezultate la fiecare rulare.
data = {
    "TransactionID": range(1, 21),
    "Amount": np.random.randint(10, 1000, 20), # sume aleatorii intre 10 si 1000
    "IsFraud": [0] * 17 + [1] * 3  # Imbalanced: 17 non-fraud (0), 3 fraud cases (3)
}

df = pd.DataFrame(data)

# Step 2: Separa datele in caracteristici si eticheta
X = df[["Amount"]]  # datele de intrare (Amount)
y = df["IsFraud"]   # eticheta pe care vrem sa o prezicem (IsFraud)

print("Original Class Distribution:", Counter(y)) # afiseaza cate exemple avem din fiecare clasa

# Step 3: Aplicarea Undersampling (reducerea clasei majoritare)
undersampler = RandomUnderSampler(
    sampling_strategy=0.5,  # vrem ca nr clasei minoritare (1) sa fie jumatate din cel al clasei majoritare (0)
    random_state=42
)
X_under, y_under = undersampler.fit_resample(X, y)

print("Class Distribution After Undersampling:", Counter(y_under))

# Step 4: Aplicarea Oversampling (SMOTE with reduced n_neighbors)
smote = SMOTE(
    sampling_strategy=0.8, # vrem ca frauda sa ajunga 80% din nr tranzactiilor normale
                           # daca avem 17 non-fraud, atunci se creeaza 13 noi exemple frauduloase (total fraud = 13)
    random_state=42,
    k_neighbors=1 # foloseste doar un vecin pentru generarea exemplelor - deoarece clasa minoritara e foarte mica (doar 3 cazuri)
)
X_smote, y_smote = smote.fit_resample(X, y)

print("Class Distribution After Oversampling (SMOTE):", Counter(y_smote))
