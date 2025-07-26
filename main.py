import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisi ROI Label Odds", layout="wide")
st.title("Analisi ROI per Label Odds (Back & Lay) + Statistiche Squadre")

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
    teams_home = ["Tutte"] + sorted(df[df["league"] == selected_league]["home_team"].dropna().unique() if selected_league != "Tutte" else df["home_team"].dropna().unique())
    selected_home = st.sidebar.selectbox("Seleziona Home Team", teams_home)

if "away_team" in df.columns:
    teams_away = ["Tutte"] + sorted(df[df["league"] == selected_league]["away_team"].dropna().unique() if selected_league != "Tutte" else df["away_team"].dropna().unique())
    selected_away = st.sidebar.selectbox("Seleziona Away Team", teams_away)

# --- APPLICA FILTRI ---
filtered_df = df[df["label_odds"] == selected_label].copy()

if selected_league != "Tutte":
    filtered_df = filtered_df[filtered_df["league"] == selected_league]

if selected_anno != "Tutti":
    filtered_df = filtered_df[filtered_df["anno"] == selected_anno]

# --- Funzione ROI (Back & Lay) ---
def calcola_roi(matches_df, segno):
    if matches_df.empty:
        return {"Matches": 0, "Win%": 0, "BackPts": 0, "LayPts": 0, "ROI%": 0, "OddMin": "-"}

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
        else:
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
    roi = round((back_pts / matches) * 100, 2) if matches > 0 else 0
    odd_min = round(100 / winrate, 2) if winrate > 0 else "-"

    return {"Matches": matches, "Win%": winrate, "BackPts": back_pts, "LayPts": lay_pts, "ROI%": roi, "OddMin": odd_min}

# --- Prospetto Back & Lay ---
league_results = []
for segno in ["HOME", "DRAW", "AWAY"]:
    league_results.append(
        ["League", segno] + list(calcola_roi(filtered_df, segno).values())
    )

# Squadra Home
if selected_home != "Tutte":
    if "Home" in selected_label:
        home_matches = filtered_df[filtered_df["home_team"] == selected_home]
    else:
        home_matches = filtered_df[filtered_df["away_team"] == selected_home]
    for segno in ["HOME", "DRAW", "AWAY"]:
        league_results.append(
            [selected_home, segno] + list(calcola_roi(home_matches, segno).values())
        )

# Squadra Away
if selected_away != "Tutte":
    if "Away" in selected_label:
        away_matches = filtered_df[filtered_df["away_team"] == selected_away]
    else:
        away_matches = filtered_df[filtered_df["home_team"] == selected_away]
    for segno in ["HOME", "DRAW", "AWAY"]:
        league_results.append(
            [selected_away, segno] + list(calcola_roi(away_matches, segno).values())
        )

results_df = pd.DataFrame(league_results, columns=["LABEL", "SEGNO", "Matches", "Win %", "Back Pts", "Lay Pts", "ROI%", "Odd Min"])

def color_rois(val):
    if isinstance(val, (int, float)) and val > 0:
        return 'background-color: #b6e8b6'
    return ''

st.subheader(f"Risultati ROI per {selected_label}")
st.dataframe(results_df.style.applymap(color_rois, subset=["Back Pts", "Lay Pts", "ROI%"]))

# --- Statistiche HT/FT ---
def mostra_statistiche_team(matches_df, squadra):
    if matches_df.empty:
        st.warning(f"Nessuna partita trovata per {squadra}.")
        return

    st.markdown(f"### Statistiche per {squadra} ({len(matches_df)} partite)")
    matches_df["risultato_ht"] = matches_df["gol_home_ht"].astype(str) + "-" + matches_df["gol_away_ht"].astype(str)
    matches_df["risultato_ft"] = matches_df["gol_home_ft"].astype(str) + "-" + matches_df["gol_away_ft"].astype(str)

    def distribuzione_risultati(col):
        risultati = matches_df[col].value_counts().reset_index()
        risultati.columns = ["Risultato", "Conteggio"]
        risultati["%"] = round((risultati["Conteggio"] / len(matches_df)) * 100, 2)
        risultati["Odd Minima"] = risultati["%"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
        return risultati

    st.write("**Risultati Esatti HT**")
    st.table(distribuzione_risultati("risultato_ht"))

    st.write("**Risultati Esatti FT**")
    st.table(distribuzione_risultati("risultato_ft"))

    # Over/Under HT
    matches_df["goals_ht"] = matches_df["gol_home_ht"] + matches_df["gol_away_ht"]
    over_ht = []
    for t in [0.5, 1.5, 2.5]:
        c = (matches_df["goals_ht"] > t).sum()
        p = round((c / len(matches_df)) * 100, 2)
        over_ht.append([f"Over {t} HT", c, p, round(100/p, 2) if p > 0 else "-"])
    st.write("**Over Goals HT**")
    st.table(pd.DataFrame(over_ht, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

    # Over/Under FT
    matches_df["goals_ft"] = matches_df["gol_home_ft"] + matches_df["gol_away_ft"]
    over_ft = []
    for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
        c = (matches_df["goals_ft"] > t).sum()
        p = round((c / len(matches_df)) * 100, 2)
        over_ft.append([f"Over {t} FT", c, p, round(100/p, 2) if p > 0 else "-"])
    st.write("**Over Goals FT**")
    st.table(pd.DataFrame(over_ft, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

    # BTTS
    btts = ((matches_df["gol_home_ft"] > 0) & (matches_df["gol_away_ft"] > 0)).sum()
    p_btts = round((btts / len(matches_df)) * 100, 2)
    odd_btts = round(100 / p_btts, 2) if p_btts > 0 else "-"
    st.write(f"**BTTS SI: {btts} ({p_btts}%) - Odd Minima: {odd_btts}**")

if selected_home != "Tutte":
    home_matches = filtered_df[filtered_df["home_team"] == selected_home]
    mostra_statistiche_team(home_matches, selected_home)

if selected_away != "Tutte":
    away_matches = filtered_df[filtered_df["away_team"] == selected_away]
    mostra_statistiche_team(away_matches, selected_away)
