import streamlit as st
import psycopg2
import pandas as pd

st.title("Filtro Completo Matches")

# Funzione connessione al database
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

# Creiamo un filtro per ogni colonna
for col in df.columns:
    if col.lower() == "id":  # ignora ID
        continue

    # Tenta di convertire in numerico (se possibile)
    try:
        df[col] = df[col].astype(str).str.replace(",", ".")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    except:
        pass

    if pd.api.types.is_numeric_dtype(df[col]):
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        selected_range = st.slider(
            f"Filtro per {col}",
            min_val, max_val, (min_val, max_val), step=0.01
        )
        filters[col] = selected_range
    else:
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) > 0:
            selected_val = st.selectbox(
                f"Filtra per {col} (opzionale)",
                ["Tutti"] + [str(v) for v in unique_vals]
            )
            if selected_val != "Tutti":
                filters[col] = selected_val

# Applica i filtri
filtered_df = df.copy()
for col, val in filters.items():
    if isinstance(val, tuple):
        filtered_df = filtered_df[(filtered_df[col] >= val[0]) & (filtered_df[col] <= val[1])]
    else:
        filtered_df = filtered_df[filtered_df[col].astype(str) == val]

st.subheader("Dati Filtrati")
st.dataframe(filtered_df)
st.write(f"**Righe visualizzate:** {len(filtered_df)}")
