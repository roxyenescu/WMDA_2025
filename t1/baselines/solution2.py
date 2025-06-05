### **Exercise 2: Web Scraping a Product Listings Page**
import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops"
response = requests.get(URL)
soup = BeautifulSoup(response.text, "html.parser")
products = []

print("Products: \n")
for product in soup.find_all("div", class_="thumbnail"):
    name = product.find("a", class_="title").text.strip()
    price = product.find("h4", class_="price").text.strip()
    products.append([name, price])

# Convertirea intr-un DataFrame
df = pd.DataFrame(products, columns=["Product Name", "Price"])

# Eliminarea duplicatelor daca exista
df_cleaned = df.drop_duplicates()

# Afisarea primelor 10 produse din datele curatate
print("Scraped Product Listings: \n")
print(df_cleaned.head(10))

# Salvarea datelor curatate intr-un CSV
df_cleaned.to_csv("scraped_products.csv", index=False)
