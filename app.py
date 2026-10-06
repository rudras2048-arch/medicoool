"""RefillClock | Modern Chronic Medicine Predictive Inventory OS.

A world-class, silky-smooth web application:
- Linear / Vercel-tier design system with Tailwind CSS & Lucide Icons
- Instant client-side state transitions (zero Streamlit reload lag)
- Interactive Refill Queue, Orbit Radar, Wholesale PO Engine, Churn Watch, Margin Arbitrage & Benchmark Defense
"""
import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="RefillClock — Pharmacy Inventory OS",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background-color: #080C14; }
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }
    iframe {
        border: none !important;
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Locate index.html
candidates = [
    os.path.join(os.path.dirname(__file__), "index.html"),
    os.path.join(os.path.dirname(__file__), "..", "index.html"),
    "/Users/rudrasharma/Desktop/hack-a-thon/index.html",
    "index.html",
]

html_path = None
for c in candidates:
    if os.path.exists(c):
        html_path = c
        break

if html_path and os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=1010, scrolling=False)
else:
    st.error("index.html not found. Please build the web application first.")
