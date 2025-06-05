## Exercise 1 (10 minutes): Load & Preprocess Your Dataset
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Load the Iris dataset from scikit-learn
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Introduce some artificial missing values (optional, for demonstration)
#    Here, we'll set a few entries to NaN in the 'petal length (cm)' column
df.iloc[5:10, 2] = np.nan

# 3. Handle missing values
#    We'll use SimpleImputer to replace NaNs with the mean of each column
imputer = SimpleImputer(strategy="mean")  # Define imputer
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=iris.feature_names) # fit_transform învață media pe coloane și înlocuiește valorile lipsă

# 4. Scale the data
#    StandardScaler transforms each feature to have mean=0 and std=1
scaler = StandardScaler() # Define scaler
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=iris.feature_names)
# StandardScaler aduce fiecare coloană la media 0 și dev. std. 1

# 5. Check the results
print("Preprocessed Dataset (scaled and imputed):\n")
print(df_scaled.head())

# 6. (Optional) Print the first few rows to confirm preprocessing
print("\nFirst 5 rows of preprocessed (scaled & imputed) data:")
print(df_scaled.head())