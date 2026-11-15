"""
Dupe finder: find products with a similar ingredient profile, ranked by price.

How it works
------------
1. Each product becomes a "bag of ingredients": its key ingredients (counted
   twice, because they define the product), its partial INCI list and its skin
   concerns (counted half, so products with the same purpose rank higher).
2. Ingredient names are normalised and synonyms merged
   (e.g. "Sodium Hyaluronate" -> "hyaluronic acid", "Tocopherol" -> "vitamin e").
3. TF-IDF weighting: ingredients found in almost every product (water, glycerin)
   count for little, rarer actives (retinol, niacinamide) count for a lot.
4. Cosine similarity between the vectors gives a 0-100 % match score.

Only numpy and pandas are used, so it runs anywhere the dashboard runs.
"""
import re
import numpy as np
import pandas as pd

# Common INCI / marketing names that mean the same thing
SYNONYMS = {
    "sodium hyaluronate": "hyaluronic acid",
    "hyaluronate": "hyaluronic acid",
    "ha": "hyaluronic acid",
    "tocopherol": "vitamin e",
    "tocopheryl acetate": "vitamin e",
    "ascorbic acid": "vitamin c",
    "ascorbyl glucoside": "vitamin c",
    "sodium ascorbyl phosphate": "vitamin c",
    "ethyl ascorbic acid": "vitamin c",
    "panthenol": "vitamin b5",
    "pro-vitamin b5": "vitamin b5",
    "provitamin b5": "vitamin b5",
    "niacinamide": "niacinamide",
    "vitamin b3": "niacinamide",
    "retinyl palmitate": "retinoid",
    "retinal": "retinoid",
    "retinaldehyde": "retinoid",
    "retinol": "retinoid",
    "bakuchiol": "bakuchiol",
    "ceramide np": "ceramides",
    "ceramide ap": "ceramides",
    "ceramide eop": "ceramides",
    "ceramide": "ceramides",
    "salicylic acid": "bha",
    "glycolic acid": "aha",
    "lactic acid": "aha",
    "mandelic acid": "aha",
    "butyrospermum parkii butter": "shea butter",
    "shea": "shea butter",
    "aloe barbadensis leaf juice": "aloe",
    "aloe vera": "aloe",
    "centella asiatica": "centella",
    "cica": "centella",
    "madecassoside": "centella",
    "zinc oxide": "mineral uv filter",
    "titanium dioxide": "mineral uv filter",
    "aqua": "water",
    "eau": "water",
    "parfum": "fragrance",
}

# Words that carry no ingredient meaning
_NOISE = {"", "and", "etc", "others", "other", "more", "complex"}


def normalise(name: str) -> str:
    """Lower-case, strip punctuation/dots and map synonyms."""
    n = re.sub(r"\.{2,}|…", "", str(name)).strip().lower()
    n = re.sub(r"\d+(\.\d+)?\s*%", "", n)          # "vitamin c 15%" -> "vitamin c"
    n = re.sub(r"\s+", " ", re.sub(r"[()*]", "", n)).strip()
    return SYNONYMS.get(n, n)


def _split(cell) -> list[str]:
    if pd.isna(cell):
        return []
    items = [normalise(x) for x in re.split(r"[,;/]", str(cell))]
    return [i for i in items if i not in _NOISE and not i.startswith("spf")]


def product_features(row) -> dict[str, float]:
    """Weighted bag of features for one product."""
    feats: dict[str, float] = {}
    def add(key, w):
        feats[key] = feats.get(key, 0.0) + w
    for ing in _split(row.get("Key Ingredients")):
        add(ing, 2.0)
    for ing in _split(row.get("INCI (partial)")):
        add(ing, 1.0)
    for c in _split(row.get("Skin Concern")):
        add("concern:" + c, 0.5)
    return feats


class DupeFinder:
    """Build once from the product table, then query as often as you like."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        bags = [product_features(r) for _, r in self.df.iterrows()]
        self.vocab = sorted({k for b in bags for k in b})
        idx = {k: i for i, k in enumerate(self.vocab)}

        tf = np.zeros((len(bags), len(self.vocab)))
        for r, bag in enumerate(bags):
            for k, w in bag.items():
                tf[r, idx[k]] = w

        # Smoothed IDF: rare ingredients matter more than ubiquitous ones
        doc_freq = (tf > 0).sum(axis=0)
        idf = np.log((1 + len(bags)) / (1 + doc_freq)) + 1
        vec = tf * idf
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vectors = vec / norms
        self.bags = bags

    def label(self, i: int) -> str:
        r = self.df.iloc[i]
        return f"{r['Brand']} — {r['Product Name']}"

    def shared_ingredients(self, i: int, j: int) -> list[str]:
        a = {k for k in self.bags[i] if not k.startswith("concern:")}
        b = {k for k in self.bags[j] if not k.startswith("concern:")}
        return sorted(a & b - {"water"})

    def find(self, i: int, top_n: int = 5, same_type: bool = True,
             cheaper_only: bool = True, min_similarity: float = 0.2,
             price_col: str = "Price_EUR", ppm_col: str = "Price_per_ml") -> pd.DataFrame:
        """Return the top_n most similar products to product i."""
        sims = self.vectors @ self.vectors[i]
        target = self.df.iloc[i]
        cand = self.df.assign(Similarity=sims)
        cand = cand[cand.index != i]
        cand = cand[cand["Similarity"] >= min_similarity]
        # skip other sizes / editions of the very same product
        cand = cand[~((cand["Brand"] == target["Brand"]) &
                      (cand["Product Name"] == target["Product Name"]))]
        if same_type:
            cand = cand[cand["Product Type"] == target["Product Type"]]

        # Compare on price per ml when both sides have it, else on price
        use_ppm = ppm_col in cand and pd.notna(target.get(ppm_col))
        basis = ppm_col if use_ppm else price_col
        if cheaper_only and pd.notna(target.get(basis)):
            cand = cand[cand[basis] < target[basis]]

        cand = cand.sort_values(["Similarity", basis], ascending=[False, True]).head(top_n)
        if pd.notna(target.get(basis)) and target[basis]:
            cand["Saving %"] = ((1 - cand[basis] / target[basis]) * 100).round(0)
        else:
            cand["Saving %"] = np.nan
        cand["Shared ingredients"] = [", ".join(self.shared_ingredients(i, j)) for j in cand.index]
        cand["Compared on"] = "price per ml" if use_ppm else "price"
        return cand
