import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.title("Filtro Completo Matches + Winrate & ROI 1X2")

# Funzione di connessione al DB
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

# Calcola esito (1/X/2)
def determina_esito(row):
    if row["gol_home_ft"] > row["gol_away_ft"]:
        return "1"
    elif row["gol_home_ft"] < row["gol_away_ft"]:
        return "2"
    else:
        return "X"

df["esito"] = df.apply(determina_esito, axis=1)

filters = {}

# Colonne gol da filtrare con menu a tendina
gol_columns_dropdown = ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht"]

for col in df.columns:
    if col.lower() == "id" or "minutaggio" in col.lower() or col.lower() == "data" or \
       any(keyword in col.lower() for keyword in ["primo", "secondo", "terzo", "quarto", "quinto"]):
        continue

    if col in gol_columns_dropdown:
        # Pulizia valori: int, no duplicati
        unique_vals = sorted(set(pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)))
        if 0 not in unique_vals:
            unique_vals = [0] + unique_vals
        selected_val = st.selectbox(
            f"Filtra per {col}",
            ["Tutti"] + [str(v) for v in unique_vals]
        )
        if selected_val != "Tutti":
            filters[col] = int(selected_val)
    else:
        # Slider per numerici
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
            # Menu a tendina per testuali
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

# --- Calcolo Winrate e ROI ---
if len(filtered_df) > 0:
    outcomes = ["1", "X", "2"]
    risultati = []

    for esito, quota_col in zip(outcomes, ["odd_home", "odd_draw", "odd_away"]):
        if quota_col not in filtered_df.columns:
            continue

        n_bets = len(filtered_df)
        n_win = (filtered_df["esito"] == esito).sum()
        winrate = (n_win / n_bets * 100) if n_bets > 0 else 0

        odds = pd.to_numeric(
            filtered_df[quota_col].astype(str).str.replace(",", "."),
            errors="coerce"
        ).fillna(0)

        profit = 0
        for idx, row in filtered_df.iterrows():
            if row["esito"] == esito:
                profit += (odds.loc[idx] - 1)
            else:
                profit -= 1

        roi = (profit / n_bets * 100) if n_bets > 0 else 0

        risultati.append({
            "Esito": esito,
            "N. Bets": n_bets,
            "WinRate %": round(winrate, 2),
            "Profit (1%)": round(profit, 2),
            "ROI %": round(roi, 2)
        })

    if risultati:
        st.subheader("WinRate & ROI (1% stake)")
        st.dataframe(pd.DataFrame(risultati))
else:
    st.warning("Nessuna partita trovata con i filtri selezionati.")
