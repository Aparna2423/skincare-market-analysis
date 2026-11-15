"""
Skincare assistant: ask for products in plain English.

Retrieval-augmented generation (RAG) over the product table:
  1. UNDERSTAND  - pull hard constraints out of the question with simple rules
                   (max price, product type, skin type, fragrance-free, vegan, ...)
                   and detect "dupe" questions ("cheaper alternative to X").
  2. RETRIEVE    - filter the products on those constraints, then rank them by
                   TF-IDF similarity between the question and each product's text
                   (or use the DupeFinder for dupe questions).
  3. GENERATE    - give the top products to an open-source LLM, which writes the
                   answer using ONLY those products. Backends, tried in order:
                     * Groq (free hosted Llama) if GROQ_API_KEY is set
                     * Ollama on this computer, if it is running
                     * no LLM: a plain list of the products found
"""
import os
import re
import difflib
import numpy as np
import pandas as pd
import requests

from dupe_finder import DupeFinder

# ── 1. Understanding the question ────────────────────────────────────────────
PRODUCT_TYPES = {
    "Moisturiser": ["moisturiser", "moisturizer", "moisturising cream", "day cream", "face cream", "night cream"],
    "Serum": ["serum"],
    "Sunscreen": ["sunscreen", "sun cream", "sunblock", "spf"],
    "Cleanser": ["cleanser", "face wash", "cleansing"],
    "Toner": ["toner"],
    "Eye Cream": ["eye cream"],
    "Exfoliant": ["exfoliant", "exfoliator", "peel"],
    "Face Oil": ["face oil", "facial oil"],
    "Lip Balm": ["lip balm"],
    "Body Lotion": ["body lotion"],
    "Foundation": ["foundation"],
    "Shampoo": ["shampoo"],
}
SKIN_TYPES = {
    "Sensitive": ["sensitive"], "Dry": ["dry"], "Oily": ["oily", "greasy"],
    "Acne-Prone": ["acne", "breakout", "spots", "blemish"], "Mature": ["mature", "ageing", "aging", "wrinkle"],
    "Combination": ["combination"], "Dehydrated": ["dehydrated"],
}
CLAIMS = {
    "Fragrance-Free": ["fragrance-free", "fragrance free", "unscented", "no fragrance", "without fragrance", "no perfume"],
    "Vegan": ["vegan"], "Cruelty-Free": ["cruelty-free", "cruelty free", "not tested on animals"],
    "Organic": ["organic"], "Natural": ["natural"],
}
DUPE_WORDS = r"\b(dupes?|alternatives? (?:to|for)|similar to|instead of|replacement for|cheaper (?:version|option) of|cheaper than (?!€|eur|\d))"


def parse_query(q: str, known_ingredients: list[str]) -> dict:
    ql = q.lower()
    out = {"max_price": None, "type": None, "skin": [], "claims": [], "no_alcohol": False,
           "ingredients": [], "dupe": bool(re.search(DUPE_WORDS, ql))}

    m = re.search(r"(?:under|below|less than|max(?:imum)?|cheaper than|up to|<)\s*(?:€|eur|euro)?\s*(\d+(?:[.,]\d+)?)", ql) \
        or re.search(r"(\d+(?:[.,]\d+)?)\s*(?:€|eur|euros?)\s*(?:or less|max)", ql)
    if m:
        out["max_price"] = float(m.group(1).replace(",", "."))

    for t, words in PRODUCT_TYPES.items():
        if any(re.search(rf"\b{re.escape(w)}", ql) for w in words):
            out["type"] = t
            break
    out["skin"] = [s for s, words in SKIN_TYPES.items() if any(re.search(rf"\b{w}", ql) for w in words)]
    out["claims"] = [c for c, words in CLAIMS.items() if any(w in ql for w in words)]
    out["no_alcohol"] = bool(re.search(r"(no|without|free of|alcohol[- ]free).{0,12}alcohol|alcohol[- ]free", ql))
    out["ingredients"] = [i for i in known_ingredients if len(i) > 3 and re.search(rf"\b{re.escape(i.lower())}\b", ql)]
    return out


# ── 2. Retrieval ─────────────────────────────────────────────────────────────
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zà-ÿ0-9]+", str(text).lower())


class ProductSearch:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.dupes = DupeFinder(self.df)
        self.labels = (self.df["Brand"] + " " + self.df["Product Name"]).tolist()
        self.known_ingredients = (self.df["Key Ingredients"].dropna().str.split(",").explode()
                                  .str.strip().loc[lambda s: s.str.len() > 0].unique().tolist())

        # TF-IDF over a text description of every product
        text_cols = ["Brand", "Product Name", "Product Type", "Skin Type", "Skin Concern",
                     "Free-From / Claims", "Key Ingredients", "Segment"]
        docs = [_tokens(" ".join(str(r.get(c, "")) for c in text_cols)) for _, r in self.df.iterrows()]
        self.vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        tf = np.zeros((len(docs), len(self.vocab)))
        for r, d in enumerate(docs):
            for w in d:
                tf[r, self.vocab[w]] += 1
        self.idf = np.log((1 + len(docs)) / (1 + (tf > 0).sum(0))) + 1
        v = tf * self.idf
        self.vectors = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def _query_vec(self, q):
        v = np.zeros(len(self.vocab))
        for w in _tokens(q):
            if w in self.vocab:
                v[self.vocab[w]] += 1
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n else v

    def find_product(self, q: str):
        """Product mentioned in the question (for dupe questions), or None.
        A product counts as mentioned if its brand appears and most of its name words do,
        or if almost all of a longer product name appears without the brand."""
        qtok = set(_tokens(q))
        generic = {"the", "de", "la", "le", "and", "for", "with", "cream", "serum", "spf", "face", "skin"}
        best, best_score = None, 0.0
        for i in range(len(self.df)):
            brand = set(_tokens(self.df.at[i, "Brand"])) - generic
            name = [w for w in _tokens(self.df.at[i, "Product Name"]) if w not in generic and len(w) > 1]
            if not name:
                continue
            cover = sum(w in qtok for w in name) / len(name)
            brand_hit = bool(brand) and brand <= qtok
            score = (cover + (0.6 if brand_hit else 0)) if (brand_hit or (cover >= 0.8 and len(name) >= 2)) else 0
            score += len(name) * 0.01               # prefer the more specific product on ties
            if score > best_score:
                best, best_score = i, score
        return best if best_score >= 0.9 else None

    def search(self, q: str, k: int = 5) -> tuple[pd.DataFrame, dict, str]:
        """Return (products, parsed constraints, note about what was done)."""
        p = parse_query(q, self.known_ingredients)

        if p["dupe"]:
            i = self.find_product(q)
            if i is not None:
                t = self.df.iloc[i]
                label = f"{t['Brand']} {t['Product Name']} (€{t['Price_EUR']:.0f}, {t['Size']})"
                # strict first, then looser: products with signature ingredients may have no close twin
                attempts = [(0.2, True, ""),
                            (0.05, True, " - no close ingredient twin, showing the nearest matches"),
                            (0.2, False, " - it is already among the cheapest, showing similar products at any price")]
                for min_sim, cheaper, how in attempts:
                    res = self.dupes.find(i, top_n=50, same_type=True, cheaper_only=cheaper, min_similarity=min_sim)
                    if p["max_price"]:
                        res = res[res["Price_EUR"] <= p["max_price"]]
                    for c in p["claims"]:
                        res = res[res["Free-From / Claims"].str.contains(c.split("-")[0], case=False, na=False)]
                    if not res.empty:
                        return res.head(k), p, f"dupe search for {label}{how}"
                return res, p, f"dupe search for {label} - no cheaper product of the same type shares its key ingredients"

        d = self.df.copy()
        d["Score"] = self.vectors @ self._query_vec(q)
        hard = []
        if p["max_price"] is not None: hard.append(("price", d["Price_EUR"] <= p["max_price"]))
        if p["type"]:                   hard.append(("type", d["Product Type"].str.contains(p["type"], case=False, na=False)))
        for c in p["claims"]:           hard.append((c, d["Free-From / Claims"].str.contains(c.split("-")[0], case=False, na=False)))
        if p["no_alcohol"]:             hard.append(("no alcohol", ~d["INCI (partial)"].str.contains("alcohol denat", case=False, na=False)))
        for s in p["skin"]:             hard.append((s, d["Skin Type"].str.contains(s + "|All", case=False, na=False)))
        for ing in p["ingredients"]:    hard.append((ing, (d["Key Ingredients"] + " " + d["INCI (partial)"]).str.contains(re.escape(ing), case=False, na=False)))

        # Apply constraints; if nothing is left, relax the least important ones (from the end)
        relaxed = []
        while True:
            mask = np.ones(len(d), dtype=bool)
            for _, m in hard:
                mask &= m.fillna(False).to_numpy()
            if mask.sum() or not hard:
                break
            relaxed.append(hard.pop()[0])
        res = d[mask].copy()
        # rank: text relevance first, rating as tie-breaker
        res["Rank"] = res["Score"] + 0.05 * res["Avg Rating"].fillna(res["Avg Rating"].mean())
        res = res.sort_values("Rank", ascending=False).head(k)
        note = "search" + (f" (relaxed: {', '.join(relaxed)} - no exact matches)" if relaxed else "")
        return res, p, note


# ── 3. Generation ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a friendly, honest skincare shopping assistant. Recommend products ONLY from the "
    "list you are given; never invent products, prices or claims. Mention brand, product name and "
    "price, and say briefly why each fits the request (ingredients, skin type, claims). If the list "
    "doesn't fully match the request, say so. Keep it under 150 words. You are not a dermatologist: "
    "for skin conditions suggest seeing one."
)


def products_as_context(res: pd.DataFrame) -> str:
    lines = []
    for _, r in res.iterrows():
        bits = [f"{r['Brand']} - {r['Product Name']}", f"type: {r['Product Type']}",
                f"price: EUR {r['Price_EUR']:.0f} for {r['Size']}" if pd.notna(r.get("Price_EUR")) else "price: unknown",
                f"rating: {r['Avg Rating']}", f"skin type: {r['Skin Type']}", f"concerns: {r['Skin Concern']}",
                f"claims: {r['Free-From / Claims']}", f"key ingredients: {r['Key Ingredients']}"]
        if "Similarity" in r and pd.notna(r["Similarity"]):
            bits.append(f"ingredient match: {r['Similarity'] * 100:.0f}%, saving per ml: {r['Saving %']:.0f}%")
        lines.append("- " + "; ".join(str(b) for b in bits))
    return "\n".join(lines)


def _chat(url, headers, model, messages, timeout):
    r = requests.post(url, headers=headers, timeout=timeout,
                      json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 400})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def generate(question: str, res: pd.DataFrame, note: str, groq_key: str | None = None,
             ollama_model: str | None = None) -> tuple[str, str]:
    """Return (answer, backend name)."""
    if res.empty:
        return ("I couldn't find products matching that. Try a broader request, e.g. "
                "*\"moisturiser for dry skin\"* or *\"cheaper alternative to La Mer\"*."), "none"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Request: {question}\n\nWhat was searched: {note}\n\nProducts:\n{products_as_context(res)}"},
    ]
    groq_key = groq_key or os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            ans = _chat("https://api.groq.com/openai/v1/chat/completions",
                        {"Authorization": f"Bearer {groq_key}"}, model, messages, timeout=30)
            return ans, f"Llama 3.1 via Groq ({model})"
        except Exception:
            pass
    try:  # Ollama exposes an OpenAI-compatible endpoint too
        model = ollama_model or os.getenv("OLLAMA_MODEL", "qwen:4b")
        ans = _chat(os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1/chat/completions",
                    {}, model, messages, timeout=120)
        return ans, f"local Ollama ({model})"
    except Exception:
        pass

    # No LLM available: plain, honest list
    head = "Here are the best matches I found" + (f" ({note})" if note.startswith("dupe") else "") + ":"
    rows = []
    for _, r in res.iterrows():
        extra = f" · {r['Similarity'] * 100:.0f}% ingredient match, {r['Saving %']:.0f}% cheaper per ml" \
            if "Similarity" in r and pd.notna(r.get("Similarity")) else ""
        rows.append(f"- **{r['Brand']} {r['Product Name']}** · €{r['Price_EUR']:.0f} ({r['Size']}) · "
                    f"{r['Avg Rating']}★ · key ingredients: {r['Key Ingredients']}{extra}")
    return head + "\n" + "\n".join(rows), "search only (no language model connected)"
