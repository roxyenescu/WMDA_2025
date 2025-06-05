# [Extracting keywords from product reviews to improve sentiment analysis.]
import pandas as pd # folosit pentru lucrul cu date
from sklearn.feature_extraction.text import TfidfVectorizer # extrage cuvinte importante din text pe baza scorului TF-IDF

# Step 1: Crearea unui set de date cu review-uri ale unor produse
reviews = [
    "This phone has a great camera and amazing battery life.",
    "The laptop performance is very fast and smooth.",
    "Terrible customer service, very disappointing experience.",
    "The product quality is top-notch and highly recommended.",
    "The delivery was late and the packaging was damaged."
]

df = pd.DataFrame({"Review": reviews})

print("Original Product Reviews:\n")
print(df)

# Step 2: Aplicarea TF-IDF Vectorization pentru extragerea cuvintelor importante
vectorizer = TfidfVectorizer(
    stop_words="english", # ignora cuvintele comune din limba engleza (the, is, and)
    max_features=5 # pastreaza doar cele mai importante 5 cuvinte (cu scoruri TF-IDF mari)
)
tfidf_matrix = vectorizer.fit_transform(df["Review"]) # transforma fiecare recenzie intr-un vector de numere care arata cat de important e fiecare cuvant din text

# Step 3: Obtinerea cuvintelor importante (keywords)
keywords = vectorizer.get_feature_names_out() # extrage cuvintele reale (nu scorurile) pe care le-a considerat importante

print("\nExtracted Keywords from Product Reviews:\n")
print(keywords)
