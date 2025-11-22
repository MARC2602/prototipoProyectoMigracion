import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
import pickle
import json

# Configuración

st.set_page_config(page_title="Dashboard de Migración Global + Forecast",
                   layout="wide")

st.title("🌍 Dashboard de Migración Global + Predicciones LSTM")

#Archivos
@st.cache_data
def load_panel():
    return pd.read_csv("panel_dataset.csv")

@st.cache_resource
def load_model():
    return keras.models.load_model("modelo_global (3).keras")

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_country_ids():
    with open("country_to_id.json") as f:
        return json.load(f)



panel = load_panel()
model = load_model()
scaler = load_scaler()
country_to_id = load_country_ids()


window = 10


#Predicción
def predict_future_custom(df_country, model, scaler, country_to_id,
                          window, start_year, future_gdp, n_years):

    df_country = df_country.sort_values("Year").copy()
    df_cut = df_country[df_country["Year"] <= start_year].copy()

    if len(df_cut) < window:
        st.error(f"Se necesitan al menos {window} años antes de {start_year} para predecir.")
        return None

    seq_values = df_cut[["net_migration", "GDP growth (annual %)"]].values[-window:]
    country_id = country_to_id[df_country["Country Code"].iloc[0]]
    seq_ids = np.array([country_id] * window).reshape(-1, 1)

    seq = np.hstack([seq_ids, seq_values])

    preds_scaled = []

    for _ in range(n_years):
        pred = model.predict(seq.reshape(1, window, seq.shape[1]), verbose=0)[0, 0]

        next_vec = np.array([country_id, pred, future_gdp])
        seq = np.vstack([seq[1:], next_vec])

        preds_scaled.append(pred)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)

    preds_inverted = scaler.inverse_transform(
        np.hstack([preds_scaled, np.zeros((n_years, 1))])
    )[:, 0]

    last_year = start_year
    years = np.arange(last_year + 1, last_year + 1 + n_years)

    return pd.DataFrame({
        "Year": years,
        "forecast_net_migration": preds_inverted
    })


#Dashboard
st.subheader("🌎 Mapa global de migración neta")

years = sorted(panel["Year"].unique())
selected_year = st.slider("Selecciona un año:", 
                          min_value=int(min(years)),
                          max_value=int(max(years)),
                          value=2020)

df_year = panel[panel["Year"] == selected_year]

fig_map = px.choropleth(
    df_year,
    locations="Country Code",
    color="net_migration",
    hover_name="Country Name",
    hover_data={
        "Country Code": True,
        "net_migration": True,
        "GDP growth (annual %)": True,
        "Year": True
    },
    color_continuous_scale="RdYlGn_r",
    title=f"Migración neta por país ({selected_year})",
    projection="natural earth"
)

st.plotly_chart(fig_map, use_container_width=True)


#Selección de país y visualización histórica
st.subheader("📈 Tendencia histórica de migración neta")

country = st.selectbox("Selecciona un país:", sorted(panel["Country Name"].unique()))
df_country = panel[panel["Country Name"] == country]

fig_line = px.line(
    df_country,
    x="Year",
    y="net_migration",
    markers=True,
    title=f"Migración neta de {country} (histórico)"
)
st.plotly_chart(fig_line, use_container_width=True)


#Prediccion
st.subheader("Predicción personalizada basada en LSTM")

years_available = sorted(df_country["Year"].unique())
start_year = st.selectbox(
    "Año desde donde iniciar el forecast:",
    years_available[:-window]
)

last_gdp = df_country["GDP growth (annual %)"].iloc[-1]

future_gdp = st.number_input(
    "GDP Growth futuro (%):",
    value=float(last_gdp),
    step=0.1,
    help="Usado por el modelo para estimar la migración futura."
)

years_to_predict = st.slider("Años a predecir:", 1, 20, 5)


if st.button("Generar predicción personalizada"):
    forecast = predict_future_custom(df_country, model, scaler, country_to_id,
                                     window, start_year, future_gdp, years_to_predict)

    if forecast is not None:
        st.subheader("Resultados:")
        st.dataframe(forecast)

        # Gráfica real vs predicho
        df_real_cut = df_country[df_country["Year"] <= start_year]

        fig = px.line()
        fig.add_scatter(x=df_real_cut["Year"], y=df_real_cut["net_migration"],
                        mode="lines+markers", name="Real")
        fig.add_scatter(x=forecast["Year"], y=forecast["forecast_net_migration"],
                        mode="lines+markers", name="Predicción")

        fig.update_layout(
            title=f"Real vs Predicción de migración futura — {country}",
            xaxis_title="Año",
            yaxis_title="Migración neta (personas)"
        )

        st.plotly_chart(fig, use_container_width=True)


st.markdown("""
---
**Fuente:** Banco Mundial — Indicadores de Migración y Crecimiento Económico  
**Modelo:** Global LSTM con embeddings por país  
**Visualización:** Streamlit + Plotly  
""")
