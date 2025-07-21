
import streamlit as st 
import psycopg2
import pandas as pd
import numpy as np

st.title("Filtro Completo Matches (FT, HT & Analisi Rimonta)")

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

df = run_query('SELECT * FROM "Matches";')
st.write(f"**Righe totali nel dataset:** {len(df)}")

if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df.insert(
        loc=df.columns.get_loc("away_team") + 1,
        column="risultato_ft",
        value=df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)
    )

if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df.insert(
        loc=df.columns.get_loc("gol_home_ht"),
        column="risultato_ht",
        value=df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)
    )

filters = {}
gol_columns_dropdown = ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht"]

for col in df.columns:
    if col.lower() == "id" or "minutaggio" in col.lower() or col.lower() == "data" or        any(keyword in col.lower() for keyword in ["primo", "secondo", "terzo", "quarto", "quinto"]):
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
    st.table(distribuzione)

    count_1 = distribuzione[distribuzione["Risultato"].str.contains("casa vince")].Conteggio.sum() +               distribuzione[distribuzione["Risultato"].isin(["1-0","2-0","2-1","3-0","3-1","3-2"])].Conteggio.sum()
    count_2 = distribuzione[distribuzione["Risultato"].str.contains("ospite vince")].Conteggio.sum() +               distribuzione[distribuzione["Risultato"].isin(["0-1","0-2","0-3","1-2","1-3","2-3"])].Conteggio.sum()
    count_x = distribuzione[distribuzione["Risultato"].str.contains("pareggio")].Conteggio.sum() +               distribuzione[distribuzione["Risultato"].isin(["0-0","1-1","2-2","3-3"])].Conteggio.sum()

    totale = len(df)
    winrate_df = pd.DataFrame({
        "Esito": ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
        "Conteggio": [count_1, count_x, count_2],
        "WinRate %": [round((count_1/totale)*100,2), round((count_x/totale)*100,2), round((count_2/totale)*100,2)]
    })
    st.subheader(f"WinRate {titolo}")
    st.table(winrate_df)

if not filtered_df.empty:
    mostra_distribuzione(filtered_df, "risultato_ft", "Risultati Finali (FT)")
    mostra_distribuzione(filtered_df, "risultato_ht", "Risultati Primo Tempo (HT)")

# --- Analisi Vantaggio Casa ---
if not filtered_df.empty and "primo_gol_home" in filtered_df.columns and "secondo_gol_home" in filtered_df.columns:
    home_vantaggio = filtered_df[
        filtered_df["primo_gol_home"].notnull() &
        ((filtered_df["primo_gol_away"].isnull()) | (filtered_df["primo_gol_away"] > filtered_df["primo_gol_home"]))
    ]
    count_raddoppio = 0
    count_pareggio = 0
    total_cases = 0

    for _, row in home_vantaggio.iterrows():
        primo_home = row["primo_gol_home"]
        secondo_home = row["secondo_gol_home"] if pd.notnull(row["secondo_gol_home"]) else None
        primo_away = row["primo_gol_away"] if pd.notnull(row.get("primo_gol_away", None)) else None

        if secondo_home is not None:
            if primo_away is None or primo_away > secondo_home:
                count_raddoppio += 1
            elif primo_away < secondo_home:
                count_pareggio += 1
        else:
            if primo_away is not None and primo_away > primo_home:
                count_pareggio += 1

        total_cases += 1

    st.subheader("Analisi Vantaggio Casa (1-0)")
    st.write(f"**Partite analizzate:** {total_cases}")
    st.write(f"**Da 1-0 a 2-0:** {count_raddoppio} ({round((count_raddoppio/total_cases)*100, 2)}%)")
    st.write(f"**Da 1-0 a 1-1:** {count_pareggio} ({round((count_pareggio/total_cases)*100, 2)}%)")

# --- Analisi Vantaggio Trasferta ---
if not filtered_df.empty and "primo_gol_away" in filtered_df.columns and "secondo_gol_away" in filtered_df.columns:
    away_vantaggio = filtered_df[
        filtered_df["primo_gol_away"].notnull() &
        ((filtered_df["primo_gol_home"].isnull()) | (filtered_df["primo_gol_home"] > filtered_df["primo_gol_away"]))
    ]
    count_raddoppio_away = 0
    count_pareggio_home = 0
    total_cases_away = 0

    for _, row in away_vantaggio.iterrows():
        primo_away = row["primo_gol_away"]
        secondo_away = row["secondo_gol_away"] if pd.notnull(row["secondo_gol_away"]) else None
        primo_home = row["primo_gol_home"] if pd.notnull(row.get("primo_gol_home", None)) else None

        if secondo_away is not None:
            if primo_home is None or primo_home > secondo_away:
                count_raddoppio_away += 1
            elif primo_home < secondo_away:
                count_pareggio_home += 1
        else:
            if primo_home is not None and primo_home > primo_away:
                count_pareggio_home += 1

        total_cases_away += 1

    st.subheader("Analisi Vantaggio Trasferta (0-1)")
    st.write(f"**Partite analizzate:** {total_cases_away}")
    st.write(f"**Da 0-1 a 0-2:** {count_raddoppio_away} ({round((count_raddoppio_away/total_cases_away)*100, 2)}%)")
    st.write(f"**Da 0-1 a 1-1:** {count_pareggio_home} ({round((count_pareggio_home/total_cases_away)*100, 2)}%)")
