import streamlit as st 
import psycopg2
import pandas as pd
import numpy as np

st.title("Filtro Completo Matches (filtri ottimizzati)")

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

# Carica dataset
df = run_query('SELECT * FROM "Matches";')
st.write(f"**Righe totali nel dataset:** {len(df)}")

filters = {}

# Colonne gol con menu a tendina
gol_columns_dropdown = ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht"]

for col in df.columns:
    # Escludiamo colonne indesiderate
    if col.lower() == "id" or "minutaggio" in col.lower() or col.lower() == "data" or any(keyword in col.lower() for keyword in ["primo", "secondo", "terzo", "quarto", "quinto"]):
        continue

    if col in gol_columns_dropdown:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        if 0 not in unique_vals:
            unique_vals = [0] + unique_vals
        selected_val = st.selectbox(f"Filtra per {col}", ["Tutti"] + [str(v) for v in unique_vals])
        if selected_val != "Tutti":
            filters[col] = int(selected_val)
    else:
        # Se numerica, slider
        col_temp = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
        if col_temp.notnull().sum() > 0:
            min_val = col_temp.min(skipna=True)
            max_val = col_temp.max(skipna=True)
            if pd.notna(min_val) and pd.notna(max_val):
                step_val = 0.01
                selected_range = st.slider(
                    f"Filtro per {col}",
                    float(min_val), float(max_val),
                    (float(min_val), float(max_val)),
                    step=step_val
                )
                filters[col] = (selected_range, col_temp)
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
        range_vals, col_temp = val
        mask = (col_temp >= range_vals[0]) & (col_temp <= range_vals[1])
        filtered_df = filtered_df[mask.fillna(True)]
    else:
        filtered_df = filtered_df[filtered_df[col].astype(str) == str(val)]

st.subheader("Dati Filtrati")
st.dataframe(filtered_df)
st.write(f"**Righe visualizzate:** {len(filtered_df)}")

# --- Calcolo WinRate e ROI (1% stake) ---
if not filtered_df.empty:
    st.subheader("WinRate & ROI (1% stake)")
    
    # Determina l'esito (1, X, 2)
    def get_esito(row):
        if row["gol_home_ft"] > row["gol_away_ft"]:
            return "1"
        elif row["gol_home_ft"] < row["gol_away_ft"]:
            return "2"
        else:
            return "X"
    
    filtered_df["esito"] = filtered_df.apply(get_esito, axis=1)
    
    results = []
    for esito in ["1", "X", "2"]:
        n_bets = len(filtered_df)
        n_win = (filtered_df["esito"] == esito).sum()
        winrate = (n_win / n_bets) * 100 if n_bets > 0 else 0
        # ROI calcolato come (vittorie * quota - numero totale di bet) * 1%
        quota_media = pd.to_numeric(filtered_df[f"quota_{esito.lower()}"].astype(str).str.replace(",", "."), errors="coerce").mean()
        profit = (n_win * quota_media - n_bets) * 0.01 if quota_media else 0
        roi = (profit / (n_bets * 0.01)) * 100 if n_bets > 0 else 0
        
        results.append({
            "Esito": esito,
            "N. Bets": n_bets,
            "WinRate %": round(winrate, 2),
            "Profit (1%)": round(profit, 2),
            "ROI %": round(roi, 2)
        })
    
    st.table(pd.DataFrame(results))
    
    # --- Risultati esatti ---
    st.subheader("Distribuzione Risultati Esatti")
    filtered_df["risultato_esatto"] = filtered_df["gol_home_ft"].astype(str) + "-" + filtered_df["gol_away_ft"].astype(str)
    exact_results = filtered_df["risultato_esatto"].value_counts(normalize=True) * 100
    st.table(exact_results.reset_index().rename(columns={"index": "Risultato", "risultato_esatto": "Percentuale %"}))
