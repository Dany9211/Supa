import streamlit as st 
import psycopg2
import pandas as pd
import numpy as np

st.title("Filtro Completo Matches (FT, HT & First To Score)")

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

# Aggiunta colonne risultato FT e HT
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

# Funzione per distribuzione FT/HT
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

    count_1 = distribuzione[distribuzione["Risultato"].str.contains("casa vince")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["1-0","2-0","2-1","3-0","3-1","3-2"])].Conteggio.sum()
    count_2 = distribuzione[distribuzione["Risultato"].str.contains("ospite vince")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["0-1","0-2","0-3","1-2","1-3","2-3"])].Conteggio.sum()
    count_x = distribuzione[distribuzione["Risultato"].str.contains("pareggio")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["0-0","1-1","2-2","3-3"])].Conteggio.sum()

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

# FIRST TO SCORE - ignorando partite senza gol
if not filtered_df.empty and "primo_gol_home" in filtered_df.columns and "primo_gol_away" in filtered_df.columns:
    def first_to_score(row):
        home_goal = row["primo_gol_home"]
        away_goal = row["primo_gol_away"]
        if pd.isnull(home_goal) and pd.isnull(away_goal):
            return np.nan  # ignora se entrambi vuoti
        if pd.isnull(home_goal):
            return "Away"
        if pd.isnull(away_goal):
            return "Home"
        return "Home" if home_goal < away_goal else "Away"

    filtered_df["first_to_score"] = filtered_df.apply(first_to_score, axis=1)
    filtered_ft = filtered_df.dropna(subset=["first_to_score"])
    st.subheader("Distribuzione First To Score (solo partite con almeno un gol)")
    first_counts = filtered_ft["first_to_score"].value_counts().reset_index()
    first_counts.columns = ["First To Score", "Conteggio"]
    first_counts["Percentuale %"] = (first_counts["Conteggio"] / len(filtered_ft) * 100).round(2)
    st.table(first_counts)
