### **Exercise 1: Extracting and Cleaning Data from an API**
### -> EXTRAGEREA SI CURATAREA DATELOR DINTR-UN API

import requests # pentru trimiterea de cereri HTTP
import pandas as pd # pentru lucrul cu tabele

cities = ["New York", "London", "Tokyo", "Paris", "Berlin"]
weather_data = []

for city in cities:
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.text.strip()
        condition, temperature = data.rsplit(" ", 1) # extragerea conditiei de vreme si a temperaturii
        temperature = temperature.replace("°C", "") # curatarea valorilor de temperatura

        weather_data.append([city, condition, temperature])
    else:
        print(f"Failed to retrieve data for {city}")

# Convertirea in DataFrame
df = pd.DataFrame(weather_data, columns=["City", "Weather Condition", "Temperature (°C)"])

# Salvarea datelor curatate intr-un CSV
df.to_csv("cleaned_weather_data.csv", index=False)

# Afisarea dataset-ului
print(df)


