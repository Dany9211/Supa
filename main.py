import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.title("Filtro Completo Matches (con WinRate & ROI)")

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

# Crea colonna risultato_ft
df["risultato_ft"] = df["gol_home_ft"].astype(str) + " - " + df["gol_away_ft"].astype(str)

filters = {}
gol_columns_dropdown = ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht"]

# Filtri dinamici
for col in df.columns:
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

# Applica filtri
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

# Distribuzione dei risultati esatti principali
risultati_interessanti = [
    "0 - 0", "0 - 1", "0 - 2", "0 - 3",
    "1 - 0", "1 - 1", "1 - 2", "1 - 3",
    "2 - 0", "2 - 1", "2 - 2", "2 - 3",
    "3 - 0", "3 - 1", "3 - 2", "3 - 3"
]
distribuzione = filtered_df["risultato_ft"].value_counts().reset_index()
distribuzione.columns = ["Risultato", "Conteggio"]
distribuzione = distribuzione[distribuzione["Risultato"].isin(risultati_interessanti)]
distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(filtered_df) * 100).round(2)
st.subheader("Distribuzione Risultati Esatti")
st.dataframe(distribuzione)

# WinRate & ROI calcolato con quote odd_home, odd_draw, odd_away
def calcola_roi(filtered_df):
    esiti = []
    n_bets = []
    win_rates = []
    profits = []
    rois = []

    outcomes = {"1": "odd_home", "X": "odd_draw", "2": "odd_away"}
    for esito, quota_col in outcomes.items():
        n_bet = len(filtered_df)
        n_win = 0
        profit = 0

        for _, row in filtered_df.iterrows():
            if (row["gol_home_ft"] > row["gol_away_ft"] and esito == "1") or \
               (row["gol_home_ft"] < row["gol_away_ft"] and esito == "2") or \
               (row["gol_home_ft"] == row["gol_away_ft"] and esito == "X"):
                n_win += 1
                profit += (row[quota_col] - 1) * 0.01
            else:
                profit -= 0.01

        esiti.append(esito)
        n_bets.append(n_bet)
        win_rates.append(round((n_win / n_bet) * 100, 2) if n_bet > 0 else 0)
        profits.append(round(profit * 100, 2))  # convertito in punti su 1%
        rois.append(round((profit / (n_bet * 0.01)) * 100, 2) if n_bet > 0 else 0)

    return pd.DataFrame({"Esito": esiti, "N. Bets": n_bets, "WinRate %": win_rates, "Profit (1%)": profits, "ROI %": rois})

if len(filtered_df) > 0:
    st.subheader("WinRate & ROI (1% stake)")
    roi_df = calcola_roi(filtered_df)
    st.dataframe(roi_df)
