import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.title("Filtro Avanzato Matches - Tutto il Dataset")

# Connessione al database
def run_query(query):
    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        dbname=st.secrets["postgres"]["dbname"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        sslmode="require"
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Carico tutto il dataset
df = run_query('SELECT * FROM "Matches";')

st.write(f"**Righe totali:** {len(df)}")

filters = {}
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):  # Se colonna numerica
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        selected_range = st.slider(f"Filtro per {col}", min_val, max_val, (min_val, max_val))
        filters[col] = selected_range
    else:  # Se colonna testuale/categoriale
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) > 1:
            selected_val = st.selectbox(f"Filtra per {col} (opzionale)", ["Tutti"] + [str(v) for v in unique_vals])
            if selected_val != "Tutti":
                filters[col] = selected_val

# Applico i filtri
filtered_df = df.copy()
for col, val in filters.items():
    if isinstance(val, tuple):  # Filtro range numerico
        filtered_df = filtered_df[(filtered_df[col] >= val[0]) & (filtered_df[col] <= val[1])]
    else:  # Filtro stringa
        filtered_df = filtered_df[filtered_df[col].astype(str) == val]

st.subheader("Dati Filtrati")
st.dataframe(filtered_df)
st.write(f"**Righe visualizzate:** {len(filtered_df)}")
