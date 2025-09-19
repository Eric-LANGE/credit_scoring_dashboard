import streamlit as st
import requests
import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from io import BytesIO
from PIL import Image

# --- Constants ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SHAP_IMAGE_PATH = PROJECT_ROOT / "shap" / "shap_beeswarm.png"
PLOTS_DIR = PROJECT_ROOT / "plots"

st.set_page_config(layout="wide", page_title="Dashboard Score Crédit")

API_URL = "http://127.0.0.1:8000"


# --- API & Data Communication Functions ---


@st.cache_data(ttl=3600)
def get_customer_ids():
    """Fetches the list of all customer IDs from the API."""
    try:
        response = requests.get(f"{API_URL}/customers")
        response.raise_for_status()
        return response.json().get("customer_ids", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        return []


def get_api_data_for_customer(customer_id):
    """Fetches all necessary data from the new, specific API endpoints."""
    try:
        score_res = requests.get(f"{API_URL}/customer/{customer_id}/score")
        features_res = requests.get(f"{API_URL}/customer/{customer_id}/features")
        shap_res = requests.get(f"{API_URL}/customer/{customer_id}/shap")

        score_res.raise_for_status()
        features_res.raise_for_status()
        shap_res.raise_for_status()

        return {
            "score_data": score_res.json(),
            "features": features_res.json(),
            "shap_values": shap_res.json(),
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de charger les données pour le client {customer_id}: {e}")
        return None


@st.cache_data(ttl=3600)
def get_bivariate_data_from_api(feat_x, feat_y):
    """Fetches data for the bi-variate scatter plot."""
    try:
        response = requests.get(
            f"{API_URL}/features/bivariate_data?feat_x={feat_x}&feat_y={feat_y}"
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


@st.cache_data(ttl=3600)
def load_distribution_data(feature_name):
    """Loads histogram data directly from the static JSON file."""
    file_path = PLOTS_DIR / f"{feature_name}_hist_data.json"
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# --- Image & Layout Stabilization Helpers ---
# Pistes appliquées : container/placeholder + dimensions fixes + cache bytes
STABLE_BOX_H_SMALL = 260  # px (metrics/plots compacts)
STABLE_BOX_H_MED = 380  # px (plots matplotlib)
IMG_WIDTH_SMALL = 360  # px
IMG_WIDTH_MED = 720  # px


@st.cache_data(show_spinner=False)
def load_image_file_bytes(path_str: str) -> bytes:
    p = Path(path_str)
    with open(p, "rb") as f:
        return f.read()


def stable_image_box_begin(height_px: int):
    st.markdown(
        f"""
        <style>
        .stable-image-box {{
            display:flex;align-items:center;justify-content:center;
            height:{height_px}px;overflow:hidden;
        }}
        /* Optionnel : fige (approximativement) la largeur de la sidebar pour limiter les reflows */
        section[data-testid="stSidebar"] > div:first-child {{
            min-width: 300px; max-width: 300px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="stable-image-box">', unsafe_allow_html=True)


def stable_image_box_end():
    st.markdown("</div>", unsafe_allow_html=True)


# --- Plot Generation Functions ---
def create_shap_waterfall_plot(plot_data):
    if plot_data is None:
        return None
    base_value = plot_data["base_value"]
    shap_values_np = np.array(plot_data["values"])
    feature_names = plot_data["feature_names"]
    explanation = shap.Explanation(
        values=shap_values_np, base_values=base_value, feature_names=feature_names
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig = plt.gcf()
    return fig


def create_bivariate_plot(plot_data, customer_features, feat_x, feat_y):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_data["x_data"], plot_data["y_data"], alpha=0.1, color="grey")
    if (
        customer_features.get(feat_x) is not None
        and customer_features.get(feat_y) is not None
    ):
        ax.scatter(
            customer_features[feat_x],
            customer_features[feat_y],
            color="red",
            s=150,
            edgecolor="black",
            zorder=5,
        )
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    return fig


def create_distribution_plot(counts, bin_edges, median_val, client_value, feature_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        align="edge",
        color="lightgrey",
        edgecolor="grey",
    )
    if pd.notna(median_val):
        ax.axvline(
            median_val,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:,.2f}",
        )
    if client_value is not None:
        ax.axvline(
            client_value,
            color="red",
            linestyle="--",
            lw=3,
            label=f"Client : {client_value:.2f}",
        )
    ax.legend()
    ax.set_xlabel(feature_name, fontsize=11)
    ax.set_ylabel("Frequency (Count)", fontsize=11)
    return fig


# --- UI Display Functions ---


def display_score_and_features(api_data, customer_id):
    # **FIX:** Add customer ID to the header
    st.header(f"Analyse Client : {customer_id}")
    score_data = api_data.get("score_data")
    customer_features = api_data.get("features")
    score = (1 - score_data["probability_pos"]) * 100

    col1, col2 = st.columns([1.15, 1])
    with col1:
        st.subheader("Score de Crédit")
        gauge_threshold = (1 - score_data["threshold"]) * 100
        bar_color = "green" if score_data["decision"] == "accepted" else "red"
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": f"Seuil : {gauge_threshold:.2f}%", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": bar_color},
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": gauge_threshold,
                    },
                },
            )
        )
        fig_gauge.update_layout(margin=dict(l=20, r=20, t=70, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("Caractéristiques Principales")
        html_metrics = ""
        for feature, value in customer_features.items():
            if feature in ["DAYS_EMPLOYED", "OWN_CAR_AGE"] and pd.notna(value):
                formatted_value = f"{int(value)}"
            elif isinstance(value, float):
                formatted_value = f"{value:.3f}"
            elif pd.isna(value):
                formatted_value = "N/A"
            else:
                formatted_value = str(value)

            html_metrics += f"""
            <div style="margin-bottom: 12px;">
                <div style="font-size: 0.9rem; color: #808495;">{feature}</div>
                <div style="font-size: 2.25rem; font-weight: 600;">{formatted_value}</div>
            </div>
            """
        st.markdown(html_metrics, unsafe_allow_html=True)


def display_shap_importance(api_data):
    st.header("Contribution des Caractéristiques au Score")
    col_g, col_l = st.columns([1.15, 1])
    with col_g:
        st.subheader("Importance Globale")
        # --- Stable box + width fixe (anti reflow/resize) ---
        stable_image_box_begin(STABLE_BOX_H_MED)
        placeholder_global = st.empty()
        beeswarm_bytes = load_image_file_bytes(str(SHAP_IMAGE_PATH))
        placeholder_global.image(
            Image.open(BytesIO(beeswarm_bytes)), width=IMG_WIDTH_MED
        )
        stable_image_box_end()
    with col_l:
        st.subheader("Importance Locale")
        # --- Stable box + placeholder (évite le "blink") ---
        stable_image_box_begin(STABLE_BOX_H_MED)
        plot_placeholder = st.empty()
        with st.spinner("Génération du graphique SHAP..."):
            shap_data = api_data.get("shap_values")
            if shap_data:
                fig = create_shap_waterfall_plot(shap_data)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                plt.close(fig)
                plot_placeholder.image(buf, width=IMG_WIDTH_MED)
            else:
                plot_placeholder.error("Données SHAP non disponibles.")
        stable_image_box_end()


def display_customer_positioning(customer_features):
    st.header("Positionnement du Client")
    features_to_plot = list(customer_features.keys())
    col_dist, col_bi = st.columns(2)

    with col_dist:
        st.subheader("Distribution d'une Caractéristique")
        feature_dist = st.selectbox("Caractéristique", features_to_plot, key="dist")
        if feature_dist:
            hist_data = load_distribution_data(feature_dist)
            if hist_data:
                fig_dist = create_distribution_plot(
                    np.array(hist_data["counts"]),
                    np.array(hist_data["bin_edges"]),
                    hist_data["median"],
                    customer_features.get(feature_dist),
                    feature_name=feature_dist,
                )
                stable_image_box_begin(STABLE_BOX_H_MED)
                dist_placeholder = st.empty()
                dist_placeholder.image(fig_to_bytes(fig_dist), width=IMG_WIDTH_MED)
                stable_image_box_end()
            else:
                st.error("Impossible de charger les données de distribution.")

    with col_bi:
        st.subheader("Analyse Bi-variée")
        col_x, col_y = st.columns(2)
        feat_x = col_x.selectbox("Axe X", features_to_plot, index=0, key="bx")
        feat_y = col_y.selectbox("Axe Y", features_to_plot, index=1, key="by")
        if feat_x and feat_y:
            with st.spinner("Chargement des données bi-variées..."):
                bivariate_data = get_bivariate_data_from_api(feat_x, feat_y)
            if bivariate_data:
                fig_bi = create_bivariate_plot(
                    bivariate_data, customer_features, feat_x, feat_y
                )
                stable_image_box_begin(STABLE_BOX_H_MED)
                bi_placeholder = st.empty()
                bi_placeholder.image(fig_to_bytes(fig_bi), width=IMG_WIDTH_MED)
                stable_image_box_end()
            else:
                st.error("Impossible de générer le graphique bi-varié.")


def fig_to_bytes(fig):
    """Converts a matplotlib figure to a bytes object for Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return buf.getvalue()


# --- Main Application Logic ---
st.title("Dashboard d'Analyse de Risque de Crédit")

customer_ids = get_customer_ids()
if not customer_ids:
    st.warning(
        "Liste des clients non chargée. Vérifiez que l'API est en cours d'exécution."
    )
else:
    selected_id = st.sidebar.selectbox("Sélectionnez un ID Client", customer_ids)
    if selected_id:
        with st.spinner(f"Chargement des données pour le client {selected_id}..."):
            api_data = get_api_data_for_customer(selected_id)

        if api_data:
            # **FIX:** Pass the selected_id to the display function
            display_score_and_features(api_data, selected_id)
            st.markdown("---")
            display_shap_importance(api_data)
            st.markdown("---")
            display_customer_positioning(api_data.get("features", {}))
        else:
            st.error(
                f"Impossible de récupérer les données pour le client {selected_id}."
            )
