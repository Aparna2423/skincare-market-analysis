import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skincare Product Intelligence",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #e9ecef;
    }
    .metric-label { font-size: 12px; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 24px; font-weight: 600; color: #212529; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 6px 16px;
        font-size: 14px;
    }
    h1 { font-size: 26px !important; font-weight: 600 !important; }
    h2 { font-size: 18px !important; font-weight: 500 !important; }
    h3 { font-size: 15px !important; font-weight: 500 !important; color: #495057 !important; }
    .sidebar-section { font-size: 11px; font-weight: 600; color: #6c757d;
                        text-transform: uppercase; letter-spacing: .06em;
                        margin: 1rem 0 .3rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Skincare_Products_Complete.xlsx", sheet_name="All Products")
    except FileNotFoundError:
        st.error("❌ Could not find 'Skincare_Products_Complete.xlsx'. Make sure it is in the same folder as this script.")
        st.stop()

    # Clean price — extract numeric value, treat SEK as ~EUR/10 approximation
    def parse_price(p):
        if pd.isna(p):
            return None
        p = str(p)
        val = pd.to_numeric(''.join(filter(lambda c: c.isdigit() or c == '.', p)), errors='coerce')
        if 'SEK' in p and val:
            val = round(val / 10.5, 1)   # rough SEK → EUR conversion
        return val

    df['Price_EUR'] = df['Price (approx)'].apply(parse_price)

    # Clean size → ml/g
    df['Size_ml'] = df['Size'].str.extract(r'(\d+\.?\d*)').astype(float)

    # Price per ml
    df['Price_per_ml'] = (df['Price_EUR'] / df['Size_ml']).round(3)

    # Fragrance & alcohol flags
    inci = df['INCI (partial)'].fillna('').str.lower()
    df['Contains_Fragrance'] = inci.str.contains('parfum|fragrance', na=False)
    df['Contains_Alcohol']   = inci.str.contains('alcohol denat|denatured alcohol', na=False)

    # Avg Rating as float
    df['Avg Rating'] = pd.to_numeric(df['Avg Rating'], errors='coerce')

    # Num Reviews as int
    df['Num Reviews'] = pd.to_numeric(df['Num Reviews'], errors='coerce').fillna(0).astype(int)

    # Product category derived from product type keywords
    df['Category'] = df['Product Type'].apply(_categorize_product)

    # Price tier group derived from segment label
    df['Segment_Tier'] = df['Segment'].apply(_categorize_segment)

    return df


def _categorize_product(product_type):
    if pd.isna(product_type):
        return 'Other'
    pt = str(product_type).lower().strip()

    bath_body = [
        'body lotion', 'body cream', 'body wash', 'body scrub',
        'body butter', 'body oil', 'hand cream', 'baby lotion',
        'baby cream', 'body serum'
    ]
    haircare = [
        'shampoo', 'conditioner', 'hair serum', 'hair oil',
        'hair mask', 'hair spray', 'hair mousse',
        'leave-in conditioner', 'heat protectant',
        'dry shampoo', 'hair treatment', 'hair primer'
    ]
    makeup = [
        'foundation', 'bb cream', 'cc cream', 'concealer', 'blush',
        'lipstick', 'lip gloss', 'eyeliner', 'mascara',
        'eyeshadow', 'eye shadow', 'brow', 'highlighter',
        'bronzer', 'setting powder', 'setting spray',
        'primer', 'lip plumper'
    ]
    skincare = [
        'cleanser', 'moisturiser', 'sunscreen', 'serum', 'exfoliant',
        'essence', 'toner', 'mask', 'eye cream', 'face cream',
        'face oil', 'facial mist', 'night cream', 'sheet mask',
        'cleansing oil', 'eye patch', 'night serum'
    ]

    # Check more specific/prefixed categories first to avoid false substring matches
    for kw in bath_body:
        if kw in pt:
            return 'Bath & Body'
    for kw in haircare:
        if kw in pt:
            return 'Haircare'
    for kw in makeup:
        if kw in pt:
            return 'Makeup'
    for kw in skincare:
        if kw in pt:
            return 'Skincare'
    return 'Other'


def _categorize_segment(segment):
    if pd.isna(segment):
        return 'Other'
    s = str(segment).lower().strip()
    if any(kw in s for kw in ['luxury', 'prestige', 'high-end', 'high end', 'premium']):
        return 'Luxury & Prestige'
    if any(kw in s for kw in ['professional', 'salon', 'clinical', 'derma']):
        return 'Professional'
    if any(kw in s for kw in ['natural', 'organic', 'clean', 'eco']):
        return 'Natural & Clean'
    if any(kw in s for kw in ['mid', 'moderate', 'accessible']):
        return 'Mid-Range'
    if any(kw in s for kw in ['drugstore', 'mass', 'budget', 'affordable', 'value']):
        return 'Budget-Friendly'
    return 'Other'


CONCERN_GROUPS = {
    'Acne & Breakouts':       ['acne', 'breakout', 'blemish', 'pore', 'blackhead', 'whitehead'],
    'Anti-Aging':             ['aging', 'ageing', 'wrinkle', 'fine line', 'firmness', 'sagging', 'elasticity', 'lifting'],
    'Brightening':            ['brighten', 'dark spot', 'hyperpigmentation', 'uneven', 'dull', 'pigmentation', 'glow', 'radiance'],
    'Hydration & Dryness':    ['dry', 'dehydra', 'moisture', 'hydra'],
    'Sensitivity & Redness':  ['sensitiv', 'redness', 'irritat', 'rosacea', 'eczema', 'calm', 'sooth'],
    'Eye Concerns':           ['dark circle', 'puffy', 'puffiness', 'eye bag', 'crow'],
    'Oil Control':            ['oily', 'shine', 'sebum', 'mattif'],
}

SKIN_TYPE_GROUPS = {
    'Dry Skin':       ['dry'],
    'Oily Skin':      ['oily'],
    'Combination':    ['combination'],
    'Sensitive Skin': ['sensitiv'],
    'Normal Skin':    ['normal'],
    'All Skin Types': ['all'],
}

HAIR_CONCERN_GROUPS = {
    'Damage & Repair':   ['damage', 'repair', 'breakage', 'split end', 'strengthen', 'bond'],
    'Moisture & Frizz':  ['dry', 'moisture', 'hydra', 'frizz', 'humid'],
    'Scalp Health':      ['scalp', 'dandruff', 'flake', 'itch'],
    'Volume & Lift':     ['volume', 'fine hair', 'thin hair', 'lift', 'limp'],
    'Colour Protection': ['colour', 'color', 'dye', 'bleach', 'highlight', 'tinted'],
    'Growth & Loss':     ['growth', 'hair loss', 'shedding', 'alopecia', 'stimulat'],
    'Curl Definition':   ['curl', 'wave', 'coil', 'define'],
    'Oil Control':       ['oily', 'grease', 'sebum', 'refresh'],
}

HAIR_TYPE_GROUPS = {
    'Fine Hair':      ['fine', 'thin'],
    'Thick & Coarse': ['thick', 'coarse'],
    'Curly & Wavy':   ['curl', 'wave', 'coil'],
    'Straight':       ['straight'],
    'Colour-Treated': ['colour', 'color', 'bleach', 'chemical'],
    'Dry Hair':       ['dry'],
    'Oily Hair':      ['oily', 'grease'],
    'All Hair Types': ['all'],
}

BODY_CONCERN_GROUPS = {
    'Dryness & Hydration': ['dry', 'moisture', 'hydra', 'flak'],
    'Firming & Tone':      ['firm', 'tone', 'tighten', 'elasticity', 'cellulite', 'contour'],
    'Sensitivity':         ['sensitiv', 'gentle', 'sooth', 'calm', 'irritat'],
    'Brightening':         ['brighten', 'glow', 'radiance', 'even'],
    'Anti-Aging':          ['aging', 'ageing', 'wrinkle', 'fine line'],
    'Stretch Marks':       ['stretch mark', 'scar', 'mark'],
}

_CATEGORY_CONFIG = {
    'Skincare':    {'profile_heading': 'Skin',    'concern_label': 'Skin concern',  'concern_groups': CONCERN_GROUPS,      'type_label': 'Skin type', 'type_groups': SKIN_TYPE_GROUPS},
    'Makeup':      {'profile_heading': 'Skin',    'concern_label': 'Skin concern',  'concern_groups': CONCERN_GROUPS,      'type_label': 'Skin type', 'type_groups': SKIN_TYPE_GROUPS},
    'Haircare':    {'profile_heading': 'Hair',    'concern_label': 'Hair concern',  'concern_groups': HAIR_CONCERN_GROUPS, 'type_label': 'Hair type', 'type_groups': HAIR_TYPE_GROUPS},
    'Bath & Body': {'profile_heading': 'Body',    'concern_label': 'Body concern',  'concern_groups': BODY_CONCERN_GROUPS, 'type_label': 'Skin type', 'type_groups': SKIN_TYPE_GROUPS},
    'Other':       {'profile_heading': 'Profile', 'concern_label': 'Concern',       'concern_groups': CONCERN_GROUPS,      'type_label': 'Type',      'type_groups': SKIN_TYPE_GROUPS},
}

df = load_data()

# ── Helpers: expand comma-separated fields ────────────────────────────────────
def unique_vals(col):
    vals = set()
    for v in df[col].dropna():
        for item in str(v).split(','):
            vals.add(item.strip())
    return sorted(vals)

def unique_vals_from(source_df, col):
    vals = set()
    for v in source_df[col].dropna():
        for item in str(v).split(','):
            vals.add(item.strip())
    return sorted(vals)

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("# 🧴 Beauty Explorer")
st.sidebar.markdown("---")

# ── Static: always-visible filters ───────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-section">Brand & Ingredients</div>', unsafe_allow_html=True)
brand_search = st.sidebar.text_input("Brand search", placeholder="e.g. CeraVe")
countries    = ['All'] + sorted(df['Country of Brand'].dropna().unique().tolist())
sel_country  = st.sidebar.selectbox("Country of brand", countries)
ingr_search  = st.sidebar.text_input("Key ingredient contains", placeholder="e.g. Niacinamide")
ff_only      = st.sidebar.checkbox("Fragrance-free claims only")
no_alcohol   = st.sidebar.checkbox("No alcohol denat")

st.sidebar.markdown("---")

# ── Dynamic: progressive filters ─────────────────────────────────────────────
_max_price = int(df['Price_EUR'].dropna().max()) + 1

st.sidebar.markdown('<div class="sidebar-section">Product</div>', unsafe_allow_html=True)
categories   = ['All', 'Skincare', 'Makeup', 'Haircare', 'Bath & Body', 'Other']
sel_category = st.sidebar.selectbox("Product category", categories)

# Defaults used when no category is selected
sel_type          = []
sel_tier_group    = 'All'
sel_segment       = 'All'
price_range       = (0, _max_price)
sel_concern_group = []
sel_concern       = []
sel_type_group    = []
sel_profile_type  = []

if sel_category != 'All':
    cfg  = _CATEGORY_CONFIG[sel_category]
    _pre = df[df['Category'] == sel_category].copy()

    # Product type — multiselect scoped to category
    _type_opts = sorted(_pre['Product Type'].dropna().unique().tolist())
    sel_type = st.sidebar.multiselect("Product type", _type_opts)
    if sel_type:
        _pre = _pre[_pre['Product Type'].isin(sel_type)]

    # Price — scoped to pre-filtered set
    st.sidebar.markdown('<div class="sidebar-section">Price</div>', unsafe_allow_html=True)
    _tier_opts = sorted(_pre['Segment_Tier'].dropna().unique().tolist())
    sel_tier_group = st.sidebar.selectbox("Price tier (group)", ['All'] + _tier_opts)
    if sel_tier_group != 'All':
        _seg_opts = _pre[_pre['Segment_Tier'] == sel_tier_group]['Segment'].dropna().unique().tolist()
    else:
        _seg_opts = _pre['Segment'].dropna().unique().tolist()
    sel_segment = st.sidebar.selectbox("Price tier", ['All'] + sorted(_seg_opts))
    price_range = st.sidebar.slider("Price range (EUR)", 0, _max_price, (0, _max_price))

    # Cascade tier selection into _pre before building profile dropdowns
    if sel_tier_group != 'All':
        _pre = _pre[_pre['Segment_Tier'] == sel_tier_group]
    if sel_segment != 'All':
        _pre = _pre[_pre['Segment'] == sel_segment]

    # Profile — heading and labels driven by category config
    st.sidebar.markdown(
        f'<div class="sidebar-section">{cfg["profile_heading"]} Profile</div>',
        unsafe_allow_html=True
    )

    # Concern area (group) — multiselect, options from _pre
    _raw_concerns = set(unique_vals_from(_pre, 'Skin Concern'))
    _avail_cg = [
        g for g, kws in cfg['concern_groups'].items()
        if any(any(kw.lower() in c.lower() for kw in kws) for c in _raw_concerns)
    ]
    sel_concern_group = st.sidebar.multiselect("Concern area", _avail_cg)

    if sel_concern_group:
        _grp_kws     = [kw for g in sel_concern_group for kw in cfg['concern_groups'][g]]
        _concern_pool = [
            c for c in unique_vals_from(_pre, 'Skin Concern')
            if any(kw.lower() in c.lower() for kw in _grp_kws) and len(c) < 30
        ]
    else:
        _concern_pool = [c for c in unique_vals_from(_pre, 'Skin Concern') if len(c) < 30]
    sel_concern = st.sidebar.multiselect(cfg['concern_label'], _concern_pool)

    # Type group — multiselect, options from _pre
    _raw_types = set(unique_vals_from(_pre, 'Skin Type'))
    _avail_tg  = [
        g for g, kws in cfg['type_groups'].items()
        if any(any(kw.lower() in t.lower() for kw in kws) for t in _raw_types)
    ]
    sel_type_group = st.sidebar.multiselect(f"{cfg['type_label']} (group)", _avail_tg)

    if sel_type_group:
        _tg_kws    = [kw for g in sel_type_group for kw in cfg['type_groups'][g]]
        _type_pool = [
            t for t in unique_vals_from(_pre, 'Skin Type')
            if any(kw.lower() in t.lower() for kw in _tg_kws) and len(t) < 20
        ]
    else:
        _type_pool = [t for t in unique_vals_from(_pre, 'Skin Type') if len(t) < 20]
    sel_profile_type = st.sidebar.multiselect(cfg['type_label'], _type_pool)

# ── Apply filters ─────────────────────────────────────────────────────────────
fdf = df.copy()

if sel_category != 'All':
    fdf = fdf[fdf['Category'] == sel_category]

if sel_type:
    fdf = fdf[fdf['Product Type'].isin(sel_type)]

if sel_tier_group != 'All':
    fdf = fdf[fdf['Segment_Tier'] == sel_tier_group]

if sel_segment != 'All':
    fdf = fdf[fdf['Segment'] == sel_segment]

fdf = fdf[
    (fdf['Price_EUR'].isna()) |
    ((fdf['Price_EUR'] >= price_range[0]) & (fdf['Price_EUR'] <= price_range[1]))
]

if sel_concern_group and sel_category != 'All':
    _cfg     = _CATEGORY_CONFIG[sel_category]
    _pattern = '|'.join(kw for g in sel_concern_group for kw in _cfg['concern_groups'][g])
    fdf = fdf[fdf['Skin Concern'].fillna('').str.contains(_pattern, case=False)]

if sel_concern:
    _pattern = '|'.join(re.escape(c) for c in sel_concern)
    fdf = fdf[fdf['Skin Concern'].fillna('').str.contains(_pattern, case=False)]

if sel_type_group and sel_category != 'All':
    _cfg     = _CATEGORY_CONFIG[sel_category]
    _pattern = '|'.join(kw for g in sel_type_group for kw in _cfg['type_groups'][g])
    fdf = fdf[fdf['Skin Type'].fillna('').str.contains(_pattern, case=False)]

if sel_profile_type:
    _pattern = '|'.join(re.escape(t) for t in sel_profile_type)
    fdf = fdf[fdf['Skin Type'].fillna('').str.contains(_pattern, case=False)]

if sel_country != 'All':
    fdf = fdf[fdf['Country of Brand'] == sel_country]

if brand_search:
    fdf = fdf[fdf['Brand'].str.contains(brand_search, case=False, na=False)]

if ingr_search:
    fdf = fdf[
        fdf['Key Ingredients'].fillna('').str.contains(ingr_search, case=False) |
        fdf['INCI (partial)'].fillna('').str.contains(ingr_search, case=False)
    ]

if ff_only:
    fdf = fdf[fdf['Free-From / Claims'].fillna('').str.contains('fragrance.free', case=False)]

if no_alcohol:
    fdf = fdf[~fdf['Contains_Alcohol']]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧴 Skincare Product Intelligence Dashboard")
st.caption(f"Analysing **{len(fdf):,}** products across **{fdf['Brand'].nunique()}** brands | "
           f"from a total dataset of **{len(df):,}** products")

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
avg_price = fdf['Price_EUR'].mean()
avg_rating = fdf['Avg Rating'].mean()
total_reviews = fdf['Num Reviews'].sum()
avg_ppm = fdf['Price_per_ml'].mean()

k1.metric("Products", f"{len(fdf):,}")
k2.metric("Brands", f"{fdf['Brand'].nunique()}")
k3.metric("Avg price", f"€{avg_price:.0f}" if not np.isnan(avg_price) else "—")
k4.metric("Avg rating", f"{avg_rating:.2f} ★" if not np.isnan(avg_rating) else "—")
k5.metric("Total reviews", f"{total_reviews:,}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Compare", "💰 Price Analysis", "⭐ Ratings", "🔬 Ingredients", "📋 Product Table"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Brand · Segment · Concern comparison")

    c1, c2 = st.columns(2)

    # ── Segment breakdown donut ───────────────────────────────────────────────
    with c1:
        st.markdown("#### Products by segment")
        seg_counts = fdf['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig = px.pie(
            seg_counts, names='Segment', values='Count',
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, width='stretch')

    # ── Product type breakdown ────────────────────────────────────────────────
    with c2:
        st.markdown("#### Top product types")
        type_counts = fdf['Product Type'].value_counts().head(12).reset_index()
        type_counts.columns = ['Type', 'Count']
        fig2 = px.bar(
            type_counts, x='Count', y='Type', orientation='h',
            color='Count', color_continuous_scale='Teal',
            text='Count'
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10), height=320,
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig2, width='stretch')

    st.markdown("---")

    # ── Top skin concerns heatmap ─────────────────────────────────────────────
    st.markdown("#### Skin concerns × Segment")
    concern_seg = []
    for _, row in fdf.iterrows():
        if pd.notna(row['Skin Concern']) and pd.notna(row['Segment']):
            for c in str(row['Skin Concern']).split(','):
                c = c.strip()
                if c and len(c) < 25:
                    concern_seg.append({'Concern': c, 'Segment': row['Segment']})

    if concern_seg:
        cs_df = pd.DataFrame(concern_seg)
        top_concerns = cs_df['Concern'].value_counts().head(12).index
        top_segs = cs_df['Segment'].value_counts().head(10).index
        cs_pivot = cs_df[
            cs_df['Concern'].isin(top_concerns) & cs_df['Segment'].isin(top_segs)
        ].groupby(['Concern', 'Segment']).size().unstack(fill_value=0)

        fig3 = px.imshow(
            cs_pivot,
            color_continuous_scale='Teal',
            aspect='auto',
            labels=dict(color="Products")
        )
        fig3.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=380,
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig3, width='stretch')

    # ── Country of origin breakdown ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Brands by country of origin")
    country_counts = fdf['Country of Brand'].value_counts().head(15).reset_index()
    country_counts.columns = ['Country', 'Count']
    fig4 = px.bar(
        country_counts, x='Country', y='Count',
        color='Count', color_continuous_scale='Blues',
        text='Count'
    )
    fig4.update_traces(textposition='outside')
    fig4.update_layout(
        showlegend=False, coloraxis_showscale=False,
        margin=dict(t=20, b=10, l=10, r=10), height=320
    )
    st.plotly_chart(fig4, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRICE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Price intelligence")

    c1, c2 = st.columns(2)

    # ── Box plot: price by segment ────────────────────────────────────────────
    with c1:
        st.markdown("#### Price distribution by segment")
        price_seg = fdf[fdf['Price_EUR'].notna()].copy()
        seg_order = price_seg.groupby('Segment')['Price_EUR'].median().sort_values().index.tolist()
        fig5 = px.box(
            price_seg, x='Segment', y='Price_EUR',
            category_orders={'Segment': seg_order},
            color='Segment',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={'Price_EUR': 'Price (EUR)', 'Segment': ''}
        )
        fig5.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=380,
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig5, width='stretch')

    # ── Bar: median price by product type ─────────────────────────────────────
    with c2:
        st.markdown("#### Median price by product type")
        type_price = (
            fdf[fdf['Price_EUR'].notna()]
            .groupby('Product Type')['Price_EUR']
            .agg(['median', 'count'])
            .reset_index()
        )
        type_price.columns = ['Type', 'Median_Price', 'Count']
        type_price = type_price[type_price['Count'] >= 2].sort_values('Median_Price', ascending=False).head(15)
        fig6 = px.bar(
            type_price, x='Median_Price', y='Type', orientation='h',
            color='Median_Price', color_continuous_scale='Oranges',
            labels={'Median_Price': 'Median Price (EUR)', 'Type': ''},
            text=type_price['Median_Price'].apply(lambda x: f'€{x:.0f}')
        )
        fig6.update_traces(textposition='outside')
        fig6.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10), height=380,
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig6, width='stretch')

    st.markdown("---")

    # ── Scatter: price vs rating coloured by segment ──────────────────────────
    st.markdown("#### Price vs rating — coloured by segment")
    scatter_df = fdf[fdf['Price_EUR'].notna() & fdf['Avg Rating'].notna()].copy()
    if len(scatter_df) > 0:
        fig7 = px.scatter(
            scatter_df,
            x='Price_EUR', y='Avg Rating',
            color='Segment',
            size='Num Reviews',
            size_max=30,
            hover_data=['Brand', 'Product Name', 'Product Type'],
            labels={'Price_EUR': 'Price (EUR)', 'Avg Rating': 'Average Rating'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig7.update_layout(
            margin=dict(t=20, b=10, l=10, r=10), height=420,
            legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1)
        )
        st.plotly_chart(fig7, width='stretch')

    # ── Best value: price per ml for selected type ────────────────────────────
    st.markdown("---")
    st.markdown("#### Best value — price per ml")
    col_ppm, _ = st.columns([2, 1])
    with col_ppm:
        ppm_type = st.selectbox(
            "Select product type to compare value",
            options=['All types'] + sorted(fdf['Product Type'].dropna().unique().tolist()),
            key='ppm_type'
        )

    ppm_df = fdf[fdf['Price_per_ml'].notna() & fdf['Price_EUR'].notna()].copy()
    if ppm_type != 'All types':
        ppm_df = ppm_df[ppm_df['Product Type'] == ppm_type]

    ppm_df = ppm_df.sort_values('Price_per_ml').head(20)
    if len(ppm_df) > 0:
        ppm_df['Label'] = ppm_df['Brand'] + ' — ' + ppm_df['Product Name'].str[:35]
        fig8 = px.bar(
            ppm_df, x='Price_per_ml', y='Label', orientation='h',
            color='Segment',
            labels={'Price_per_ml': 'Price per ml (EUR)', 'Label': ''},
            hover_data=['Price_EUR', 'Size'],
            text=ppm_df['Price_per_ml'].apply(lambda x: f'€{x:.2f}/ml'),
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig8.update_traces(textposition='outside')
        fig8.update_layout(
            showlegend=True,
            margin=dict(t=10, b=10, l=10, r=20),
            height=max(320, len(ppm_df) * 28),
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig8, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RATINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Rating & review analysis")

    c1, c2 = st.columns(2)

    # ── Avg rating by segment ─────────────────────────────────────────────────
    with c1:
        st.markdown("#### Average rating by segment")
        rat_seg = (
            fdf[fdf['Avg Rating'].notna()]
            .groupby('Segment')['Avg Rating']
            .mean()
            .reset_index()
            .sort_values('Avg Rating', ascending=True)
        )
        rat_seg['Avg Rating'] = rat_seg['Avg Rating'].round(2)
        overall_avg = fdf['Avg Rating'].mean()

        fig9 = go.Figure()
        fig9.add_trace(go.Bar(
            x=rat_seg['Avg Rating'],
            y=rat_seg['Segment'],
            orientation='h',
            marker_color=[
                '#27ae60' if v >= overall_avg else '#e67e22'
                for v in rat_seg['Avg Rating']
            ],
            text=rat_seg['Avg Rating'].apply(lambda x: f'{x:.2f}'),
            textposition='outside'
        ))
        fig9.add_vline(x=overall_avg, line_dash='dash', line_color='grey',
                       annotation_text=f'avg {overall_avg:.2f}', annotation_position='top right')
        fig9.update_layout(
            margin=dict(t=10, b=10, l=10, r=60), height=380,
            xaxis=dict(range=[3.8, 5.1])
        )
        st.plotly_chart(fig9, width='stretch')

    # ── Top rated products ────────────────────────────────────────────────────
    with c2:
        st.markdown("#### Top rated products (min 5,000 reviews)")
        top_rated = (
            fdf[fdf['Num Reviews'] >= 5000]
            .sort_values('Avg Rating', ascending=False)
            .head(12)[['Brand', 'Product Name', 'Avg Rating', 'Num Reviews', 'Segment', 'Price_EUR']]
        )
        if len(top_rated) > 0:
            top_rated['Stars'] = top_rated['Avg Rating'].apply(lambda r: '★' * int(round(r)) + '☆' * (5 - int(round(r))))
            top_rated['Price'] = top_rated['Price_EUR'].apply(lambda p: f'€{p:.0f}' if pd.notna(p) else '—')
            top_rated['Reviews'] = top_rated['Num Reviews'].apply(lambda n: f'{n:,}')
            st.dataframe(
                top_rated[['Brand', 'Product Name', 'Stars', 'Avg Rating', 'Reviews', 'Segment', 'Price']],
                hide_index=True,
                width='stretch',
                height=360
            )
        else:
            st.info("No products match the current filters with 5,000+ reviews.")

    st.markdown("---")

    # ── Review volume by brand ────────────────────────────────────────────────
    st.markdown("#### Most reviewed brands")
    brand_reviews = (
        fdf.groupby('Brand')['Num Reviews'].sum()
        .reset_index()
        .sort_values('Num Reviews', ascending=False)
        .head(20)
    )
    brand_reviews.columns = ['Brand', 'Total Reviews']
    fig10 = px.bar(
        brand_reviews, x='Total Reviews', y='Brand',
        orientation='h',
        color='Total Reviews', color_continuous_scale='Purples',
        text=brand_reviews['Total Reviews'].apply(lambda n: f'{n:,}')
    )
    fig10.update_traces(textposition='outside')
    fig10.update_layout(
        showlegend=False, coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=20), height=500,
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig10, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INGREDIENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Ingredient & claims analysis")

    c1, c2 = st.columns(2)

    # ── Fragrance-free claim vs INCI check ────────────────────────────────────
    with c1:
        st.markdown("#### Fragrance-free: claims vs INCI reality")
        fdf['Claims_FF'] = fdf['Free-From / Claims'].fillna('').str.lower().str.contains('fragrance.free')
        ff_df = fdf.copy()
        ff_df['Status'] = 'No claim, no parfum'
        ff_df.loc[ff_df['Claims_FF'] & ~ff_df['Contains_Fragrance'], 'Status'] = '✅ Claims FF — confirmed'
        ff_df.loc[ff_df['Claims_FF'] &  ff_df['Contains_Fragrance'], 'Status'] = '⚠️ Claims FF — parfum found'
        ff_df.loc[~ff_df['Claims_FF'] &  ff_df['Contains_Fragrance'], 'Status'] = 'No claim, has parfum'

        status_counts = ff_df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        color_map = {
            '✅ Claims FF — confirmed': '#27ae60',
            '⚠️ Claims FF — parfum found': '#e74c3c',
            'No claim, no parfum': '#95a5a6',
            'No claim, has parfum': '#f39c12'
        }
        fig11 = px.bar(
            status_counts, x='Count', y='Status', orientation='h',
            color='Status', color_discrete_map=color_map,
            text='Count'
        )
        fig11.update_traces(textposition='outside')
        fig11.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=20), height=280
        )
        st.plotly_chart(fig11, width='stretch')

        ff_claimed = ff_df[ff_df['Claims_FF']]
        if len(ff_claimed) > 0:
            mismatch_pct = ff_df[ff_df['Claims_FF'] & ff_df['Contains_Fragrance']].shape[0] / len(ff_claimed) * 100
            st.info(f"Of **{len(ff_claimed)}** products claiming fragrance-free, **{mismatch_pct:.1f}%** contain parfum in their INCI list.")

    # ── Key ingredients frequency ─────────────────────────────────────────────
    with c2:
        st.markdown("#### Most common key ingredients")
        ingr_map = {}
        for val in fdf['Key Ingredients'].dropna():
            for ingr in str(val).split(','):
                ingr = ingr.strip()
                if ingr and len(ingr) < 35:
                    ingr_map[ingr] = ingr_map.get(ingr, 0) + 1

        ingr_df = pd.DataFrame(list(ingr_map.items()), columns=['Ingredient', 'Count'])
        ingr_df = ingr_df.sort_values('Count', ascending=False).head(20)

        fig12 = px.bar(
            ingr_df, x='Count', y='Ingredient', orientation='h',
            color='Count', color_continuous_scale='Teal',
            text='Count'
        )
        fig12.update_traces(textposition='outside')
        fig12.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=20), height=500,
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig12, width='stretch')

    st.markdown("---")

    # ── Vegan / cruelty-free / organic claims ─────────────────────────────────
    st.markdown("#### Ethical & clean claims breakdown")
    claim_keywords = {
        'Vegan': 'vegan',
        'Cruelty-Free': 'cruelty.free',
        'Fragrance-Free': 'fragrance.free',
        'Paraben-Free': 'paraben.free',
        'Organic / Certified': 'organic|certified',
        'Natural': r'\bnatural\b',
        'Hypoallergenic': 'hypoallergenic',
        'Non-comedogenic': 'non.comedogenic'
    }
    claim_counts = {}
    for label, pattern in claim_keywords.items():
        count = fdf['Free-From / Claims'].fillna('').str.lower().str.contains(pattern).sum()
        claim_counts[label] = count

    claim_df = pd.DataFrame(list(claim_counts.items()), columns=['Claim', 'Products'])
    claim_df = claim_df.sort_values('Products', ascending=True)
    claim_df['Pct'] = (claim_df['Products'] / len(fdf) * 100).round(1)

    fig13 = px.bar(
        claim_df, x='Products', y='Claim', orientation='h',
        color='Pct', color_continuous_scale='Greens',
        text=claim_df['Products'].astype(str) + ' (' + claim_df['Pct'].astype(str) + '%)'
    )
    fig13.update_traces(textposition='outside')
    fig13.update_layout(
        showlegend=False, coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=40), height=340
    )
    st.plotly_chart(fig13, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PRODUCT TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Full product table")
    st.caption(f"{len(fdf):,} products match your filters")

    # Sort control
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    with sc1:
        sort_col = st.selectbox(
            "Sort by",
            ['Brand', 'Price_EUR', 'Avg Rating', 'Num Reviews', 'Product Type', 'Segment'],
            index=0
        )
    with sc2:
        sort_dir = st.radio("Order", ['Ascending', 'Descending'], horizontal=True)
    with sc3:
        show_n = st.selectbox("Show rows", [50, 100, 200, 'All'], index=0)

    sorted_df = fdf.sort_values(sort_col, ascending=(sort_dir == 'Ascending'))
    if show_n != 'All':
        sorted_df = sorted_df.head(int(show_n))

    display_cols = ['Brand', 'Product Name', 'Product Type', 'Segment',
                    'Price_EUR', 'Avg Rating', 'Num Reviews',
                    'Skin Type', 'Skin Concern', 'Key Ingredients',
                    'Country of Brand', 'Free-From / Claims']

    display_df = sorted_df[display_cols].copy()
    display_df['Price_EUR'] = display_df['Price_EUR'].apply(
        lambda p: f'€{p:.0f}' if pd.notna(p) else '—'
    )
    display_df['Avg Rating'] = display_df['Avg Rating'].apply(
        lambda r: f'{r:.1f} ★' if pd.notna(r) else '—'
    )
    display_df['Num Reviews'] = display_df['Num Reviews'].apply(
        lambda n: f'{int(n):,}' if pd.notna(n) else '—'
    )
    display_df.columns = ['Brand', 'Product', 'Type', 'Segment',
                          'Price', 'Rating', 'Reviews',
                          'Skin Type', 'Concerns', 'Key Ingredients',
                          'Country', 'Claims']

    st.dataframe(display_df, hide_index=True, width='stretch', height=520)

    # Download button
    csv = fdf.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️  Download filtered data as CSV",
        data=csv,
        file_name='skincare_filtered.csv',
        mime='text/csv'
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit · Skincare Product Intelligence Project · "
           "Dataset: 482 products, 210 brands")
