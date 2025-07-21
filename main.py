import streamlit as st
import psycopg2
import pandas as pd

st.title("Filtro Dati Matches")

# Funzione di connessione
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

# Carichiamo un campione di dati
st.info("Carico un campione di 100 righe per filtrare.")
df = run_query('SELECT * FROM "Matches" LIMIT 100;')

# Creiamo i menu a tendina per tutte le colonne
filters = {}
for col in df.columns:
    unique_vals = df[col].dropna().unique().tolist()
    if len(unique_vals) > 1:
        selected_val = st.selectbox(f"Filtra per {col} (opzionale)", ["Tutti"] + [str(v) for v in unique_vals])
        if selected_val != "Tutti":
            filters[col] = selected_val

# Applichiamo i filtri selezionati
filtered_df = df.copy()
for col, val in filters.items():
    filtered_df = filtered_df[filtered_df[col].astype(str) == val]

st.subheader("Dati filtrati")
st.dataframe(filtered_df)
st.write(f"Righe visualizzate: {len(filtered_df)}")
