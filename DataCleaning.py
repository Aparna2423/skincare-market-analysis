import pandas as pd
import numpy as np
import re

SOURCE_FILE = 'Skincare_Products_Complete.xlsx'
OUTPUT_FILE = 'moisturizer_data.csv'
SHEET_NAME = 'All Products'

PRICE_REGEX = re.compile(r'(\d+(?:[\.,]\d+)?)')
UNIT_CONVERSIONS = {
    'ml': 1.0,
    'milliliter': 1.0,
    'millilitre': 1.0,
    'g': 1.0,
    'gram': 1.0,
    'grams': 1.0,
    'oz': 29.5735,
    'fl oz': 29.5735,
    'floz': 29.5735,
}

def parse_price(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = PRICE_REGEX.search(text)
    if not match:
        return np.nan
    numeric = match.group(1).replace(',', '.')
    try:
        return float(numeric)
    except ValueError:
        return np.nan


def parse_size_ml(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()

    quantity_match = re.search(r'(\d+(?:[\.,]\d+)?)\s*[x×]\s*(\d+(?:[\.,]\d+)?)(?:\s*)(ml|milliliter|millilitre|g|gram|grams|oz|fl oz|floz)', text)
    if quantity_match:
        qty = float(quantity_match.group(1).replace(',', '.'))
        size = float(quantity_match.group(2).replace(',', '.'))
        unit = quantity_match.group(3)
        return qty * size * UNIT_CONVERSIONS.get(unit, 1.0)

    for unit, factor in UNIT_CONVERSIONS.items():
        if unit in text:
            match = PRICE_REGEX.search(text)
            if match:
                value_num = float(match.group(1).replace(',', '.'))
                return value_num * factor

    match = PRICE_REGEX.search(text)
    if match:
        return float(match.group(1).replace(',', '.'))

    return np.nan


def contains_fragrance(text):
    if pd.isna(text):
        return False
    scent_keywords = r'parfum|fragrance|aroma|linalool|limonene|citral|eugenol'
    return bool(re.search(scent_keywords, str(text).lower()))


def show_quality_report(df):
    total = len(df)
    report = {
        'total_rows': total,
        'missing_price': int(df['Price_Clean'].isna().sum()),
        'missing_size': int(df['Size_ml'].isna().sum()),
        'missing_price_per_ml': int(df['Price_per_ml'].isna().sum()),
        'zero_size': int((df['Size_ml'] == 0).sum()),
        'infinite_price_per_ml': int(np.isinf(df['Price_per_ml']).sum()),
        'missing_inci': int(df['INCI (partial)'].isna().sum()),
        'fragrance_count': int(df['Contains_Fragrance'].sum()),
    }
    print('\nData quality report:')
    for key, value in report.items():
        print(f'  {key}: {value}')


if __name__ == '__main__':
    df = pd.read_excel(SOURCE_FILE, sheet_name=SHEET_NAME)

    df['Price_Clean'] = df['Price (approx)'].apply(parse_price)
    df['Size_ml'] = df['Size'].apply(parse_size_ml)
    df['Price_per_ml'] = df['Price_Clean'] / df['Size_ml']
    df['Contains_Fragrance'] = df['INCI (partial)'].apply(contains_fragrance)
    df['Currency'] = df['Price (approx)'].str.extract(r'([A-Z]{3})')

    show_quality_report(df)

    print('\nSample cleaned rows:')
    print(df[['Brand', 'Product Name', 'Segment', 'Price (approx)', 'Size', 'Price_Clean', 'Size_ml', 'Price_per_ml', 'Contains_Fragrance']].head(15).to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f'\nSaved cleaned data to {OUTPUT_FILE}')