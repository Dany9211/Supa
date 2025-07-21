import streamlit as st
import pandas as pd
import psycopg2

# Connessione al database
@st.cache_data
def run_query(query):
    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        dbname=st.secrets["postgres"]["dbname"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"]
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

st.title("Analisi ROI Back e Lay - Football")

# Query dataset
query = "SELECT * FROM Matches"
df = run_query(query)

# Calcolo colonna "esito" (1, X, 2)
def calcola_esito(row):
    if row["gol_home_ft"] > row["gol_away_ft"]:
        return "1"
    elif row["gol_home_ft"] < row["gol_away_ft"]:
        return "2"
    else:
        return "X"

df["esito"] = df.apply(calcola_esito, axis=1)

# Filtri di base
st.sidebar.header("Filtri")

# Filtro campionato
if "league" in df.columns:
    campionati = ["Tutti"] + sorted(df["league"].dropna().unique().tolist())
    campionato = st.sidebar.selectbox("Seleziona Campionato", campionati)
    if campionato != "Tutti":
        df = df[df["league"] == campionato]

# Filtro squadre
home_teams = ["Tutte"] + sorted(df["home_team"].dropna().unique().tolist())
away_teams = ["Tutte"] + sorted(df["away_team"].dropna().unique().tolist())

home_filter = st.sidebar.selectbox("Seleziona Squadra Home", home_teams)
away_filter = st.sidebar.selectbox("Seleziona Squadra Away", away_teams)

if home_filter != "Tutte":
    df = df[df["home_team"] == home_filter]
if away_filter != "Tutte":
    df = df[df["away_team"] == away_filter]

# Filtro label odds
if "label_odds" in df.columns:
    label_odds = ["Tutte"] + sorted(df["label_odds"].dropna().unique().tolist())
    label_odds_filter = st.sidebar.selectbox("Seleziona Label Odds", label_odds)
    if label_odds_filter != "Tutte":
        df = df[df["label_odds"] == label_odds_filter]

# Mostra tabella filtrata
st.subheader("Dati Filtrati")
st.dataframe(df)

# Calcolo ROI e WinRate
def calcola_roi_winrate(df):
    risultati = []
    for esito in ["1", "X", "2"]:
        n_bets = len(df)  # puntiamo su ogni partita
        n_win = (df["esito"] == esito).sum()
        strike_rate = (n_win / n_bets * 100) if n_bets > 0 else 0

        # Calcolo profitto e ROI
        col_odds = None
        if esito == "1" and "odd_1" in df.columns:
            col_odds = "odd_1"
        elif esito == "X" and "odd_x" in df.columns:
            col_odds = "odd_x"
        elif esito == "2" and "odd_2" in df.columns:
            col_odds = "odd_2"

        profitto = 0
        if col_odds:
            for _, row in df.iterrows():
                if row["esito"] == esito:
                    profitto += (row[col_odds] - 1)  # vincita - stake
                else:
                    profitto -= 1  # stake persa
            roi = (profitto / n_bets * 100) if n_bets > 0 else 0
        else:
            roi = 0

        risultati.append({
            "Esito": esito,
            "N Bets": n_bets,
            "Win%": round(strike_rate, 2),
            "ROI %": round(roi, 2),
            "Profitto (1%)": round(profitto, 2)
        })
    return pd.DataFrame(risultati)

if len(df) > 0:
    st.subheader("ROI & WinRate")
    df_roi = calcola_roi_winrate(df)
    st.table(df_roi)
else:
    st.warning("Nessun dato corrisponde ai filtri selezionati.")
