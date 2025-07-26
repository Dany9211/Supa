import streamlit as st
import psycopg2
import pandas as pd

st.set_page_config(page_title="Analisi ROI per Label Odds", layout="wide")
st.title("Analisi ROI per Label Odds, Campionato e Squadre")

# --- Funzione connessione ---
@st.cache_data
def run_query(query: str):
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

# --- Caricamento dati ---
try:
    df = run_query('SELECT * FROM "allcamp";')
    st.write(f"**Righe nel dataset:** {len(df)}")
except Exception as e:
    st.error(f"Errore durante il caricamento: {e}")
    st.stop()

# --- Label Odds ---
def assegna_label_odds(row):
    try:
        oh = float(str(row["odd_home"]).replace(",", "."))
        oa = float(str(row["odd_away"]).replace(",", "."))
    except:
        return "N/A"

    if oh <= 1.5:
        return "Home strong fav"
    elif 1.51 <= oh <= 2.0:
        return "Home med fav"
    elif 2.01 <= oh <= 2.5 and oa > 3.0:
        return "Home small fav"
    elif oa <= 1.5:
        return "Away strong fav"
    elif 1.51 <= oa <= 2.0:
        return "Away med fav"
    elif 2.01 <= oa <= 2.5 and oh > 3.0:
        return "Away small fav"
    elif oh < 3.0 and oa < 3.0:
        return "Supercompetitive"
    return "Altro"

df["label_odds"] = df.apply(assegna_label_odds, axis=1)

# --- FILTRI ---
st.sidebar.header("Filtri")

# Filtro campionato
leagues = ["Tutti"] + sorted(df["league"].dropna().unique())
selected_league = st.sidebar.selectbox("Seleziona Campionato", leagues)

# Filtro anno
years = ["Tutti"] + sorted(df["anno"].dropna().unique())
selected_year = st.sidebar.selectbox("Seleziona Anno", years)

# Filtro label odds
labels = sorted(df["label_odds"].dropna().unique())
selected_label = st.sidebar.selectbox("Seleziona Label Odds", labels)

# Filtro squadre
teams = sorted(set(df["home_team"].dropna().unique()) | set(df["away_team"].dropna().unique()))
selected_teams = st.sidebar.multiselect("Seleziona Squadre", teams)

# --- APPLICA FILTRI ---
filtered_df = df.copy()
if selected_league != "Tutti":
    filtered_df = filtered_df[filtered_df["league"] == selected_league]
if selected_year != "Tutti":
    filtered_df = filtered_df[filtered_df["anno"] == selected_year]
if selected_label:
    filtered_df = filtered_df[filtered_df["label_odds"] == selected_label]

# --- FUNZIONE CALCOLO ROI ---
def calcola_roi(df, stake=1.0):
    if df.empty:
        return pd.DataFrame(columns=["Esito", "Winrate %", "ROI %", "Points"])

    risultati = {"1": 0, "X": 0, "2": 0}
    totale = len(df)

    for _, row in df.iterrows():
        try:
            home_goals, away_goals = int(row["gol_home_ft"]), int(row["gol_away_ft"])
            if home_goals > away_goals:
                risultati["1"] += 1
            elif home_goals == away_goals:
                risultati["X"] += 1
            else:
                risultati["2"] += 1
        except:
            continue

    data = []
    for esito in ["1", "X", "2"]:
        winrate = round(risultati[esito] / totale * 100, 2)
        profit = (risultati[esito] * (stake) - (totale - risultati[esito]) * stake)
        roi = round(profit / (totale * stake) * 100, 2)
        points = round(profit, 2)  # 1 point = 1%
        data.append([esito, winrate, roi, points])

    return pd.DataFrame(data, columns=["Esito", "Winrate %", "ROI %", "Points"])

# --- BACK e LAY ---
st.subheader(f"Analisi generale - Label: {selected_label}")
st.write(f"**Partite trovate:** {len(filtered_df)}")
st.write("**Back (1% per esito 1-X-2 su ogni partita)**")
st.table(calcola_roi(filtered_df, stake=1))

# Lay (calcolo opposto)
def calcola_lay(df, stake=1.0):
    if df.empty:
        return pd.DataFrame(columns=["Esito", "Winrate %", "ROI %", "Points"])

    data = []
    totale = len(df)

    for esito in ["1", "X", "2"]:
        perdite = 0
        vincite = 0
        for _, row in df.iterrows():
            try:
                home_goals, away_goals = int(row["gol_home_ft"]), int(row["gol_away_ft"])
                risultato = "1" if home_goals > away_goals else "X" if home_goals == away_goals else "2"
                if risultato == esito:
                    perdite += stake  # quando l'esito si verifica, perdi lo stake
                else:
                    vincite += stake
            except:
                continue
        profit = vincite - perdite
        winrate = round((vincite / (vincite + perdite)) * 100 if (vincite + perdite) > 0 else 0, 2)
        roi = round(profit / (totale * stake) * 100, 2)
        points = round(profit, 2)
        data.append([esito, winrate, roi, points])
    return pd.DataFrame(data, columns=["Esito", "Winrate %", "ROI %", "Points"])

st.write("**Lay (1% per esito 1-X-2 su ogni partita)**")
st.table(calcola_lay(filtered_df, stake=1))

# --- ANALISI PER SQUADRE SPECIFICHE ---
if selected_teams:
    st.subheader("Analisi per squadre selezionate")
    df_team = pd.DataFrame()
    for team in selected_teams:
        # Se la squadra era home con label selezionata
        df_home = filtered_df[(filtered_df["home_team"] == team)]
        # Se la squadra era away con label selezionata
        df_away = filtered_df[(filtered_df["away_team"] == team)]
        df_team = pd.concat([df_team, df_home, df_away])

    if df_team.empty:
        st.warning("Nessuna partita trovata per le squadre selezionate con questa label.")
    else:
        st.write(f"**Partite trovate per squadre selezionate:** {len(df_team)}")
        st.write("**Back (1%)**")
        st.table(calcola_roi(df_team, stake=1))
        st.write("**Lay (1%)**")
        st.table(calcola_lay(df_team, stake=1))
