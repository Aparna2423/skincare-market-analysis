import requests
import pandas as pd

url = "https://world.openbeautyfacts.org/cgi/search.pl"

params = {
    "search_terms": "moisturizer",
    "search_simple": 1,
    "action": "process",
    "json": 1,
    "page_size": 50,
    "fields": "product_name,brands,ingredients_text,categories,countries"
}

response = requests.get(url, params=params)
data = response.json()

products = data.get("products", [])
df = pd.DataFrame(products)
print(df[["product_name", "brands", "ingredients_text"]].head(10))
df.to_csv("moisturizers.csv", index=False)