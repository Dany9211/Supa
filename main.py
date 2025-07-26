import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="ROI Label Odds", layout="wide")
st.title("Analisi ROI per Label Odds")

# --- Connessione DB ---
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

try:
    df = run_query('SELECT * FROM "allcamp";')
    st.write(f"**Righe iniziali nel dataset:** {len(df)}")
except Exception as e:
    st.error(f"Errore caricamento dati: {e}")
    st.stop()

# --- Aggiungo Label Odds ---
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

# --- Filtri Sidebar ---
labels = sorted(df["label_odds"].dropna().unique())
selected_label = st.sidebar.selectbox("Seleziona Label Odds", labels)

leagues = ["Tutte"] + sorted(df["league"].dropna().unique())
selected_league = st.sidebar.selectbox("Seleziona League", leagues)

anni = ["Tutti"] + sorted(df["anno"].dropna().unique())
selected_anno = st.sidebar.selectbox("Seleziona Anno", anni)

# Filtri dinamici squadre
if selected_league != "Tutte":
    squadre_home = ["Tutte"] + sorted(df[df["league"] == selected_league]["home_team"].dropna().unique())
    squadre_away = ["Tutte"] + sorted(df[df["league"] == selected_league]["away_team"].dropna().unique())
else:
    squadre_home = ["Tutte"] + sorted(df["home_team"].dropna().unique())
    squadre_away = ["Tutte"] + sorted(df["away_team"].dropna().unique())

selected_home = st.sidebar.selectbox("Seleziona Home Team", squadre_home)
selected_away = st.sidebar.selectbox("Seleziona Away Team", squadre_away)

# --- Applica Filtri ---
filtered_df = df[df["label_odds"] == selected_label]
if selected_league != "Tutte":
    filtered_df = filtered_df[filtered_df["league"] == selected_league]
if selected_anno != "Tutti":
    filtered_df = filtered_df[filtered_df["anno"] == selected_anno]
if selected_home != "Tutte":
    filtered_df = filtered_df[filtered_df["home_team"] == selected_home]
if selected_away != "Tutte":
    filtered_df = filtered_df[filtered_df["away_team"] == selected_away]

st.subheader("Dati filtrati")
st.write(f"**Partite trovate:** {len(filtered_df)}")

# --- Calcolo ROI ---
def calcola_roi(matches_df, segno):
    if matches_df.empty:
        return {"Matches": 0, "Win %": 0, "Odd Min": "-", "Back Pts": 0, "ROI Back %": 0, "Lay Pts": 0, "ROI Lay %": 0}
    
    profitti_back, profitti_lay = [], []
    win = 0

    for _, row in matches_df.iterrows():
        home_g = int(row.get("gol_home_ft", 0))
        away_g = int(row.get("gol_away_ft", 0))
        odd_home = float(str(row.get("odd_home", 0)).replace(",", "."))
        odd_draw = float(str(row.get("odd_draw", 0)).replace(",", "."))
        odd_away = float(str(row.get("odd_away", 0)).replace(",", "."))

        if segno == "HOME":
            vincente = home_g > away_g
            odd = odd_home
        elif segno == "DRAW":
            vincente = home_g == away_g
            odd = odd_draw
        else:
            vincente = home_g < away_g
            odd = odd_away

        # Back
        profit_back = (odd - 1) if vincente else -1
        profitti_back.append(profit_back)

        # Lay
        profit_lay = - (odd - 1) if vincente else 1
        profitti_lay.append(profit_lay)

        if vincente:
            win += 1

    matches = len(matches_df)
    winrate = round((win / matches) * 100, 2) if matches > 0 else 0
    back_pts = round(sum(profitti_back), 2)
    lay_pts = round(sum(profitti_lay), 2)
    roi_back = round((back_pts / matches) * 100, 2) if matches > 0 else 0
    roi_lay = round((lay_pts / matches) * 100, 2) if matches > 0 else 0
    odd_min = round(100 / winrate, 2) if winrate > 0 else "-"

    return {"Matches": matches, "Win %": winrate, "Odd Min": odd_min, "Back Pts": back_pts, "ROI Back %": roi_back, "Lay Pts": lay_pts, "ROI Lay %": roi_lay}

# --- Tabella ROI 1/X/2 ---
roi_results = []
for segno in ["HOME", "DRAW", "AWAY"]:
    roi_results.append([segno] + list(calcola_roi(filtered_df, segno).values()))

results_df = pd.DataFrame(roi_results, columns=["Segno", "Matches", "Win %", "Odd Min", "Back Pts", "ROI Back %", "Lay Pts", "ROI Lay %"])

# --- Colorazione ROI positivi ---
def color_rois(val):
    if isinstance(val, (int, float)) and val > 0:
        return 'background-color: #b6e8b6'
    return ''

st.subheader(f"ROI per {selected_label}")
st.dataframe(results_df.style.applymap(color_rois, subset=["Back Pts", "ROI Back %", "Lay Pts", "ROI Lay %"]))
