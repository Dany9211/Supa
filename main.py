import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisi ROI Label Odds", layout="wide")
st.title("Analisi ROI per Label Odds (Back & Lay)")

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
    st.write(f"**Righe iniziali nel dataset:** {len(df)}")
except Exception as e:
    st.error(f"Errore durante il caricamento: {e}")
    st.stop()

# --- Aggiunta colonne risultato ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)
if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

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
labels = sorted(df["label_odds"].dropna().unique())
selected_label = st.sidebar.selectbox("Seleziona Label Odds", labels)

if "league" in df.columns:
    leagues = ["Tutte"] + sorted(df["league"].dropna().unique())
    selected_league = st.sidebar.selectbox("Seleziona League", leagues)

if "anno" in df.columns:
    anni = ["Tutti"] + sorted(df["anno"].dropna().unique())
    selected_anno = st.sidebar.selectbox("Seleziona Anno", anni)

# --- FILTRO SQUADRA HOME ---
if "home_team" in df.columns:
    if selected_league != "Tutte":
        home_teams = ["Tutte"] + sorted(df[df["league"] == selected_league]["home_team"].dropna().unique())
    else:
        home_teams = ["Tutte"] + sorted(df["home_team"].dropna().unique())
    selected_home = st.sidebar.selectbox("Seleziona Home Team", home_teams)

# --- FILTRO SQUADRA AWAY ---
if "away_team" in df.columns:
    if selected_league != "Tutte":
        away_teams = ["Tutte"] + sorted(df[df["league"] == selected_league]["away_team"].dropna().unique())
    else:
        away_teams = ["Tutte"] + sorted(df["away_team"].dropna().unique())
    selected_away = st.sidebar.selectbox("Seleziona Away Team", away_teams)

# --- APPLICA FILTRI ---
filtered_df = df[df["label_odds"] == selected_label].copy()
if selected_league != "Tutte":
    filtered_df = filtered_df[filtered_df["league"] == selected_league]
if selected_anno != "Tutti":
    filtered_df = filtered_df[filtered_df["anno"] == selected_anno]
if selected_home != "Tutte":
    filtered_df = filtered_df[filtered_df["home_team"] == selected_home]
if selected_away != "Tutte":
    filtered_df = filtered_df[filtered_df["away_team"] == selected_away]

# --- Funzione Calcolo ROI (Back e Lay) ---
def calcola_roi(matches_df, segno):
    if matches_df.empty:
        return {"Matches": 0, "Win%": 0, "BackPts": 0, "ROI%": 0, "LayPts": 0}

    risultati = matches_df.copy()
    risultati["back_profit"] = 0.0
    risultati["lay_profit"] = 0.0

    for i, row in risultati.iterrows():
        home_g = int(row.get("gol_home_ft", 0))
        away_g = int(row.get("gol_away_ft", 0))
        odd_home = float(str(row.get("odd_home", "0")).replace(",", "."))
        odd_draw = float(str(row.get("odd_draw", "0")).replace(",", "."))
        odd_away = float(str(row.get("odd_away", "0")).replace(",", "."))

        if segno == "HOME":
            vinta = home_g > away_g
            odd = odd_home
        elif segno == "DRAW":
            vinta = home_g == away_g
            odd = odd_draw
        else:  # AWAY
            vinta = home_g < away_g
            odd = odd_away

        risultati.at[i, "back_profit"] = (odd - 1) if vinta else -1
        risultati.at[i, "lay_profit"] = - (odd - 1) if vinta else 1

    matches = len(risultati)
    winrate = round((risultati["back_profit"] > 0).mean() * 100, 2)
    back_pts = round(risultati["back_profit"].sum(), 2)
    lay_pts = round(risultati["lay_profit"].sum(), 2)
    roi = round((back_pts / matches) * 100, 2) if matches > 0 else 0

    return {"Matches": matches, "Win%": winrate, "BackPts": back_pts, "ROI%": roi, "LayPts": lay_pts}

# --- Tabella ROI ---
league_results = []
for segno in ["HOME", "DRAW", "AWAY"]:
    league_results.append(
        ["League", segno] + list(calcola_roi(filtered_df, segno).values())
    )

results_df = pd.DataFrame(league_results, columns=["LABEL", "SEGNO", "Matches", "Win %", "Back Pts", "ROI%", "Lay Pts"])

def color_rois(val):
    if isinstance(val, (int, float)) and val > 0:
        return 'background-color: #b6e8b6'
    return ''

st.subheader(f"Risultati ROI per {selected_label}")
st.dataframe(results_df.style.applymap(color_rois, subset=["Back Pts", "Lay Pts", "ROI%"]))

# --- TOP 10 TEAM ---
def top10_teams(df, label_type):
    if df.empty:
        return pd.DataFrame()
    teams = df["home_team"].unique() if "Home" in label_type else df["away_team"].unique()
    data = []
    for team in teams:
        matches = df[(df["home_team"] == team) | (df["away_team"] == team)]
        roi_data = calcola_roi(matches, "HOME")
        data.append([team, roi_data["Matches"], roi_data["Win%"], roi_data["BackPts"], roi_data["ROI%"], roi_data["LayPts"]])
    return pd.DataFrame(data, columns=["Team", "Matches", "Win %", "Back Pts", "ROI%", "Lay Pts"]).sort_values(by="Back Pts", ascending=False).head(10)

st.subheader(f"Top 10 Team (Back Pts) per {selected_label}")
top10_df = top10_teams(filtered_df, selected_label)
if not top10_df.empty:
    for team in top10_df["Team"]:
        if st.button(f"Dettagli: {team}"):
            st.session_state["team_selected"] = team
    st.dataframe(top10_df.style.applymap(color_rois, subset=["Back Pts", "Lay Pts", "ROI%"]))
