import streamlit as st 
import psycopg2
import pandas as pd
import numpy as np

st.title("Filtro Completo Matches (FT, HT, BTTS & Over Goals con Odd Minima)")

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

# --- Aggiungi colonna risultato_ft ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df.insert(
        loc=df.columns.get_loc("away_team") + 1,
        column="risultato_ft",
        value=df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)
    )

# --- Aggiungi colonna risultato_ht ---
if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df.insert(
        loc=df.columns.get_loc("gol_home_ht"),
        column="risultato_ht",
        value=df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)
    )

filters = {}
gol_columns_dropdown = ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht"]

# --- FILTRI ---
for col in df.columns:
    if col.lower() == "id" or "minutaggio" in col.lower() or col.lower() == "data" or \
       any(keyword in col.lower() for keyword in ["primo", "secondo", "terzo", "quarto", "quinto"]):
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

# --- APPLICA FILTRI ---
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


# --- FUNZIONE DISTRIBUZIONE & WINRATE ---
def mostra_distribuzione(df, col_risultato, titolo):
    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]

    def classifica_risultato(ris):
        home, away = map(int, ris.split("-"))
        if ris in risultati_interessanti:
            return ris
        if home > away:
            return "Altro risultato casa vince"
        elif home < away:
            return "Altro risultato ospite vince"
        else:
            return "Altro pareggio"

    df[f"{col_risultato}_classificato"] = df[col_risultato].apply(classifica_risultato)

    st.subheader(f"Distribuzione {titolo}")
    distribuzione = df[f"{col_risultato}_classificato"].value_counts().reset_index()
    distribuzione.columns = ["Risultato", "Conteggio"]
    distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df) * 100).round(2)
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    st.table(distribuzione)

    # WinRate 1X2
    count_1 = distribuzione[distribuzione["Risultato"].str.contains("casa vince")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["1-0","2-0","2-1","3-0","3-1","3-2"])].Conteggio.sum()
    count_2 = distribuzione[distribuzione["Risultato"].str.contains("ospite vince")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["0-1","0-2","0-3","1-2","1-3","2-3"])].Conteggio.sum()
    count_x = distribuzione[distribuzione["Risultato"].str.contains("pareggio")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["0-0","1-1","2-2","3-3"])].Conteggio.sum()

    totale = len(df)
    winrate = [
        round((count_1/totale)*100,2),
        round((count_x/totale)*100,2),
        round((count_2/totale)*100,2)
    ]
    odd_minime = [round(100/w,2) if w>0 else "-" for w in winrate]

    winrate_df = pd.DataFrame({
        "Esito": ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
        "Conteggio": [count_1, count_x, count_2],
        "WinRate %": winrate,
        "Odd Minima": odd_minime
    })
    st.subheader(f"WinRate {titolo}")
    st.table(winrate_df)

# --- DISTRIBUZIONI FT & HT ---
if not filtered_df.empty:
    mostra_distribuzione(filtered_df, "risultato_ft", "Risultati Finali (FT)")
    mostra_distribuzione(filtered_df, "risultato_ht", "Risultati Primo Tempo (HT)")

# --- CALCOLO BTTS & OVER GOALS ---
if not filtered_df.empty:
    temp_ft = filtered_df["risultato_ft"].str.split("-", expand=True).astype(int)
    filtered_df["home_g_ft"] = temp_ft[0]
    filtered_df["away_g_ft"] = temp_ft[1]
    filtered_df["tot_goals_ft"] = filtered_df["home_g_ft"] + filtered_df["away_g_ft"]

    total_games = len(filtered_df)

    # BTTS
    btts_count = len(filtered_df[(filtered_df["home_g_ft"] > 0) & (filtered_df["away_g_ft"] > 0)])
    perc_btts = round(btts_count / total_games * 100, 2) if total_games > 0 else 0
    odd_btts = round(100 / perc_btts, 2) if perc_btts > 0 else "-"
    btts_df = pd.DataFrame([[btts_count, perc_btts, odd_btts]], columns=["Conteggio", "Percentuale %", "Odd Minima"])
    st.subheader("BTTS (Both Teams To Score)")
    st.table(btts_df)

    # Over Goals FT
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
    over_data = []
    for t in thresholds:
        count_over = (filtered_df["tot_goals_ft"] > t).sum()
        perc_over = round(count_over / total_games * 100, 2) if total_games > 0 else 0
        odd_min = round(100 / perc_over, 2) if perc_over > 0 else "-"
        over_data.append([f"Over FT {t}", count_over, perc_over, odd_min])
    over_df = pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    st.subheader("Over Goals (FT)")
    st.table(over_df)

    # Over Goals HT
    temp_ht = filtered_df["risultato_ht"].str.split("-", expand=True).astype(int)
    filtered_df["home_g_ht"] = temp_ht[0]
    filtered_df["away_g_ht"] = temp_ht[1]
    filtered_df["tot_goals_ht"] = filtered_df["home_g_ht"] + filtered_df["away_g_ht"]

    over_ht_data = []
    for t in thresholds:
        count_over_ht = (filtered_df["tot_goals_ht"] > t).sum()
        perc_over_ht = round(count_over_ht / total_games * 100, 2) if total_games > 0 else 0
        odd_min_ht = round(100 / perc_over_ht, 2) if perc_over_ht > 0 else "-"
        over_ht_data.append([f"Over HT {t}", count_over_ht, perc_over_ht, odd_min_ht])
    over_ht_df = pd.DataFrame(over_ht_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    st.subheader("Over Goals (HT)")
    st.table(over_ht_df)
