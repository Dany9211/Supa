import streamlit as st
import psycopg2
import pandas as pd

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
    if col.lower() == "id":  # Ignora ID
        continue

    # Se colonna contiene "odd" o valori numerici decimali
    if "odd" in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        selected_range = st.slider(
            f"Filtro per {col} (Quota)", 
            min_val, max_val, (min_val, max_val), step=0.01
        )
        filters[col] = selected_range
    else:
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) > 1 and len(unique_vals) <= 100:  # Evita liste troppo lunghe
            selected_val = st.selectbox(f"Filtra per {col} (opzionale)", ["Tutti"] + [str(v) for v in unique_vals])
            if selected_val != "Tutti":
                filters[col] = selected_val

# Applico i filtri
filtered_df = df.copy()
for col, val in filters.items():
    if isinstance(val, tuple):  # Range numerico
        filtered_df = filtered_df[(filtered_df[col] >= val[0]) & (filtered_df[col] <= val[1])]
    else:
        filtered_df = filtered_df[filtered_df[col].astype(str) == val]

st.subheader("Dati Filtrati")
st.dataframe(filtered_df)
st.write(f"**Righe visualizzate:** {len(filtered_df)}")
