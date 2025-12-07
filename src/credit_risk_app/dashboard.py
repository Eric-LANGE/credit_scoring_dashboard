import streamlit as st
import requests
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import base64

# --- Constants ---
st.set_page_config(layout="wide", page_title="Dashboard Score Crédit")

API_URL = "http://127.0.0.1:8000"

# --- Global font sizing (Matplotlib) ---
GLOBAL_FONT_SIZE = 12
plt.rcParams.update(
    {
        "font.size": GLOBAL_FONT_SIZE,
        "axes.titlesize": GLOBAL_FONT_SIZE + 1,
        "axes.labelsize": GLOBAL_FONT_SIZE,
        "xtick.labelsize": GLOBAL_FONT_SIZE,
        "ytick.labelsize": GLOBAL_FONT_SIZE,
        "legend.fontsize": GLOBAL_FONT_SIZE,
    }
)


# --- API & data communication functions ---
@st.cache_data(ttl=3600)
def get_customer_ids():
    try:
        response = requests.get(f"{API_URL}/customers")
        response.raise_for_status()
        return response.json().get("customer_ids", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        return []


def get_api_data_for_customer(customer_id):
    """
    Fetch all dashboard data for a customer using the composite endpoint.
    """
    try:
        response = requests.get(f"{API_URL}/customer/{customer_id}/dashboard")
        response.raise_for_status()
        data = response.json()

        return {
            "score_data": data["score"],
            "features": data["features"],
            "shap_values": data["shap"],
            "metadata": data.get("metadata"),
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de charger les données pour le client {customer_id}: {e}")
        return None


@st.cache_data(ttl=3600)
def get_bivariate_data_from_api(feat_x, feat_y):
    try:
        response = requests.get(
            f"{API_URL}/features/bivariate_data?feat_x={feat_x}&feat_y={feat_y}"
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


@st.cache_data(ttl=86400)
def get_distribution_data_from_api(feature_name):
    """Fetch histogram distribution data from API."""
    try:
        response = requests.get(f"{API_URL}/features/{feature_name}/distribution")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


@st.cache_data(ttl=86400)
def get_global_shap_image():
    """Fetch global SHAP beeswarm plot from API."""
    try:
        response = requests.get(f"{API_URL}/shap/global")
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load global SHAP plot: {e}")
        return None


# --- Image & layout stabilization helpers ---
def _inject_stable_css_once():
    if st.session_state.get("_stable_css_injected"):
        return
    st.session_state["_stable_css_injected"] = True
    st.markdown(
        """
        <style>
        .stable-image-box {
          position: relative;
          width: 100%;
          overflow: hidden;
          display: block;
        }
        .stable-image-box > img, .stable-image-box img, .stable-image-box .stable-img {
          position: absolute; inset: 0;
          width: 100%; height: 100%;
          object-fit: contain;
        }
        /* Variantes de taille (hauteur responsive + ratio)
           NOTE: on supprime le 'vh' pour éviter le décalage au tout 1er rendu
                 (iframe/viewport pas stabilisé) */
        .stable-image-box.small {
          /* Hauteur pilotée par le ratio; bornes px pour la stabilité init */
          min-height: 200px;
          max-height: 380px;
          aspect-ratio: 16 / 7;
        }
        .stable-image-box.medium {
          min-height: 220px;
          max-height: 520px;
          aspect-ratio: 16 / 9;
        }
        .stable-image-box.large {
          min-height: 240px;
          max-height: 560px;
          aspect-ratio: 2 / 1;
          /* léger décalage visuel à droite + occupation de l'espace */
          width: 97%;
          margin-left: 3%;
        }
        @media (min-width: 1000px) {
          section[data-testid="stSidebar"] > div:first-child {
            min-width: 300px; max-width: 300px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_png_in_stable_box(img_bytes: bytes, size: str = "medium"):
    """
    Rend l'image à l'intérieur de la boîte stable en un seul bloc,
    pour éviter la fragmentation DOM au 1er rendu.
    """
    _inject_stable_css_once()
    if size not in {"small", "medium", "large"}:
        size = "medium"
    b64 = base64.b64encode(img_bytes).decode("ascii")
    st.markdown(
        f"""
        <div class="stable-image-box {size}">
          <img class="stable-img" alt="plot" src="data:image/png;base64,{b64}"/>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Plot generation functions ---
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
    fig, ax = plt.subplots(figsize=(12.8, 7.2))  # 16:9
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
    fig, ax = plt.subplots(figsize=(12.8, 7.2))  # 16:9
    ax.bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        align="edge",
        color="lightgrey",
        edgecolor="grey",
    )
    is_integer_feature = feature_name in {"DAYS_EMPLOYED", "OWN_CAR_AGE"}

    def _fmt_val(v):
        if is_integer_feature:
            try:
                return f"{int(round(v))}"
            except Exception:
                return f"{v}"
        else:
            try:
                return f"{v:,.2f}"
            except Exception:
                return f"{v}"

    if pd.notna(median_val):
        ax.axvline(
            median_val,
            color="black",
            linestyle="dashdot",
            linewidth=2,
            label=f"Median: {_fmt_val(median_val)}",
        )
    if client_value is not None:
        ax.axvline(
            client_value,
            color="red",
            linestyle="solid",
            lw=3,
            label=f"Client : {_fmt_val(client_value)}",
        )
    ax.legend()
    ax.set_xlabel(feature_name, fontsize=11)
    ax.set_ylabel("Frequency (Count)", fontsize=11)
    return fig


def create_matplotlib_gauge(value: float, threshold: float, decision: str):
    value = max(0, min(100, float(value)))
    threshold = max(0, min(100, float(threshold)))

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")

    center = (0, 0)
    R_outer = 1.0
    R_inner = 0.72
    color_bar = "green" if decision == "accepted" else "red"

    from matplotlib.patches import Wedge

    # Segment de valeur
    start_angle = 180 - (value / 100.0) * 180.0
    val = Wedge(
        center,
        R_outer,
        start_angle,
        180,
        width=(R_outer - R_inner),
        facecolor=color_bar,
        edgecolor=color_bar,
    )
    ax.add_patch(val)

    th_angle = 180 - (threshold / 100.0) * 180.0
    import numpy as _np

    th_rad = _np.deg2rad(th_angle)
    x_out, y_out = R_outer * _np.cos(th_rad), R_outer * _np.sin(th_rad)
    x_in, y_in = R_inner * _np.cos(th_rad), R_inner * _np.sin(th_rad)
    ax.plot([x_in, x_out], [y_in, y_out], color="black", lw=3)

    # Libellé du seuil en 2 lignes, avec plus d'espace par rapport à la barre
    r_label = R_inner + (R_outer - R_inner) * 0.90
    tx = r_label * _np.cos(th_rad)
    ty = r_label * _np.sin(th_rad)
    ax.text(
        tx,
        ty + 0.12,  # "Seuil" au-dessus
        "Seuil",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#555555",
    )
    ax.text(
        tx,
        ty + 0.07,  # puis la valeur juste en dessous
        f"{threshold:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#555555",
    )

    # Valeur centrale
    ax.text(
        0,
        -0.10,
        f"{value:.1f}%",
        ha="center",
        va="center",
        fontsize=30,
        color="#4a5568",
    )

    for t in [0, 20, 40, 60, 80, 100]:
        a = 180 - (t / 100.0) * 180.0
        r0, r1 = R_outer * 0.96, R_outer
        ar = _np.deg2rad(a)
        ax.plot(
            [r0 * _np.cos(ar), r1 * _np.cos(ar)],
            [r0 * _np.sin(ar), r1 * _np.sin(ar)],
            color="#9e9e9e",
            lw=1,
        )
        tx_t = (R_outer * 1.06) * _np.cos(ar)
        ty_t = (R_outer * 1.06) * _np.sin(ar)
        ax.text(
            tx_t, ty_t, f"{t}", ha="center", va="center", fontsize=12, color="#6b7280"
        )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.25, 1.2)
    fig.tight_layout(pad=0.5)
    return fig


# --- UI display functions ---
def display_score_and_features(api_data, customer_id):
    st.header(f"Client : {customer_id}")
    score_data = api_data.get("score_data")
    customer_features = api_data.get("features")
    score = (1 - score_data["probability_pos"]) * 100

    col1, col2 = st.columns([1.15, 1])
    with col1:
        st.subheader("Score de crédit")
        gauge_threshold = (1 - score_data["threshold"]) * 100
        fig_g = create_matplotlib_gauge(
            value=score, threshold=gauge_threshold, decision=score_data["decision"]
        )
        render_png_in_stable_box(fig_to_bytes(fig_g), size="large")

    with col2:
        st.subheader("Caractéristiques principales")
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
    st.header("Contribution des caractéristiques au score")
    col_g, col_l = st.columns([1.15, 1])
    with col_g:
        st.subheader("Importance globale")
        beeswarm_bytes = get_global_shap_image()
        if beeswarm_bytes:
            render_png_in_stable_box(beeswarm_bytes, size="medium")
        else:
            st.error("Impossible de charger le graphique SHAP global.")
    with col_l:
        st.subheader("Importance locale")
        with st.spinner("Génération du graphique SHAP..."):
            shap_data = api_data.get("shap_values")
            if shap_data:
                fig = create_shap_waterfall_plot(shap_data)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
                plt.close(fig)
                render_png_in_stable_box(buf.getvalue(), size="medium")
            else:
                st.error("Données SHAP non disponibles.")


def display_customer_positioning(customer_features):
    st.header("Positionnement du client")
    features_to_plot = list(customer_features.keys())
    col_dist, col_bi = st.columns(2)

    with col_dist:
        st.subheader("Distribution d'une caractéristique")
        feature_dist = st.selectbox("Caractéristique", features_to_plot, key="dist")
        if feature_dist:
            hist_data = get_distribution_data_from_api(feature_dist)
            if hist_data:
                fig_dist = create_distribution_plot(
                    np.array(hist_data["counts"]),
                    np.array(hist_data["bin_edges"]),
                    hist_data["median"],
                    customer_features.get(feature_dist),
                    feature_name=feature_dist,
                )
                render_png_in_stable_box(fig_to_bytes(fig_dist), size="medium")
            else:
                st.error("Impossible de charger les données de distribution.")

    with col_bi:
        st.subheader("Analyse bi-variée")
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
                render_png_in_stable_box(fig_to_bytes(fig_bi), size="medium")
            else:
                st.error("Impossible de générer le graphique bi-varié.")


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return buf.getvalue()


# --- Main application logic ---
st.title("Dashboard d'analyse de risque de crédit")

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
            display_score_and_features(api_data, selected_id)
            st.markdown("---")
            display_shap_importance(api_data)
            st.markdown("---")
            display_customer_positioning(api_data.get("features", {}))
        else:
            st.error(
                f"Impossible de récupérer les données pour le client {selected_id}."
            )
