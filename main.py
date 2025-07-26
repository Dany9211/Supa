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

if "home_team" in df.columns:
    home_teams = ["Tutte"] + sorted(df["home_team"].dropna().unique())
    selected_home = st.sidebar.selectbox("Seleziona Home Team", home_teams)

if "away_team" in df.columns:
    away_teams = ["Tutte"] + sorted(df["away_team"].dropna().unique())
    selected_away = st.sidebar.selectbox("Seleziona Away Team", away_teams)

# --- APPLICA FILTRI ---
filtered_df = df[df["label_odds"] == selected_label].copy()

if selected_league != "Tutte":
    filtered_df = filtered_df[filtered_df["league"] == selected_league]

if selected_anno != "Tutti":
    filtered_df = filtered_df[filtered_df["anno"] == selected_anno]

# --- Funzione Calcolo ROI (Back e Lay) ---
def calcola_roi(matches_df, segno):
    if matches_df.empty:
        return {"Matches": 0, "Win%": 0, "BackPts": 0, "ROI Back %": 0, "LayPts": 0, "ROI Lay %": 0, "Odd Min": "-"}

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

        # BACK
        risultati.at[i, "back_profit"] = (odd - 1) if vinta else -1
        # LAY
        risultati.at[i, "lay_profit"] = - (odd - 1) if vinta else 1

    matches = len(risultati)
    winrate = round((risultati["back_profit"] > 0).mean() * 100, 2)
    back_pts = round(risultati["back_profit"].sum(), 2)
    lay_pts = round(risultati["lay_profit"].sum(), 2)

    roi_back = round((back_pts / matches) * 100, 2) if matches > 0 else 0
    roi_lay = round((lay_pts / matches) * 100, 2) if matches > 0 else 0
    odd_min = round(100 / winrate, 2) if winrate > 0 else "-"

    return {
        "Matches": matches,
        "Win%": winrate,
        "BackPts": back_pts,
        "ROI Back %": roi_back,
        "LayPts": lay_pts,
        "ROI Lay %": roi_lay,
        "Odd Min": odd_min
    }

# --- Tabella ROI per League, Home, Away ---
league_results = []
for segno in ["HOME", "DRAW", "AWAY"]:
    league_results.append(
        ["League", segno] + list(calcola_roi(filtered_df, segno).values())
    )

if selected_home != "Tutte":
    home_matches = filtered_df[filtered_df["home_team"] == selected_home]
    for segno in ["HOME", "DRAW", "AWAY"]:
        league_results.append(
            [selected_home, segno] + list(calcola_roi(home_matches, segno).values())
        )

if selected_away != "Tutte":
    away_matches = filtered_df[filtered_df["away_team"] == selected_away]
    for segno in ["HOME", "DRAW", "AWAY"]:
        league_results.append(
            [selected_away, segno] + list(calcola_roi(away_matches, segno).values())
        )

results_df = pd.DataFrame(league_results, columns=["LABEL", "SEGNO", "Matches", "Win %", "Back Pts", "ROI Back %", "Lay Pts", "ROI Lay %", "Odd Min"])

# --- Colorazione ROI positivi ---
def color_rois(val):
    if isinstance(val, (int, float)) and val > 0:
        return 'background-color: #b6e8b6'
    return ''

st.subheader(f"Risultati ROI per {selected_label}")
st.dataframe(results_df.style.applymap(color_rois, subset=["Back Pts", "ROI Back %", "Lay Pts", "ROI Lay %"]))

# --- TOP 10 TEAM (per Back Pts) ---
def top10_teams(df, segno="HOME"):
    teams = df["home_team"].unique()
    data = []
    for team in teams:
        matches = df[df["home_team"] == team]
        roi_data = calcola_roi(matches, segno)
        data.append([team, roi_data["Matches"], roi_data["Win%"], roi_data["BackPts"], roi_data["ROI Back %"], roi_data["LayPts"], roi_data["ROI Lay %"], roi_data["Odd Min"]])
    top_df = pd.DataFrame(data, columns=["Team", "Matches", "Win %", "Back Pts", "ROI Back %", "Lay Pts", "ROI Lay %", "Odd Min"])
    return top_df.sort_values(by="Back Pts", ascending=False).head(10)

st.subheader(f"Top 10 Team (Back Pts) per {selected_label}")
top10_df = top10_teams(filtered_df)
st.dataframe(top10_df.style.applymap(color_rois, subset=["Back Pts", "ROI Back %", "Lay Pts", "ROI Lay %"]))
