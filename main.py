import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisi ROI e Statistiche", layout="wide")
st.title("Analisi ROI per Label Odds, Squadre e Campionati")

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

# --- Caricamento dati ---
try:
    df = run_query('SELECT * FROM "allcamp";')
    st.write(f"**Righe iniziali nel dataset:** {len(df)}")
except Exception as e:
    st.error(f"Errore durante il caricamento: {e}")
    st.stop()

# --- Creazione colonne risultato ---
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
else:
    selected_league = "Tutte"

if "anno" in df.columns:
    anni = ["Tutti"] + sorted(df["anno"].dropna().unique())
    selected_anno = st.sidebar.selectbox("Seleziona Anno", anni)
else:
    selected_anno = "Tutti"

if "home_team" in df.columns:
    home_teams = ["Tutte"] + sorted(df["home_team"].dropna().unique())
    selected_home = st.sidebar.selectbox("Seleziona Home Team", home_teams)
else:
    selected_home = "Tutte"

if "away_team" in df.columns:
    away_teams = ["Tutte"] + sorted(df["away_team"].dropna().unique())
    selected_away = st.sidebar.selectbox("Seleziona Away Team", away_teams)
else:
    selected_away = "Tutte"

# --- Applica Filtri ---
filtered_df = df[df["label_odds"] == selected_label].copy()

if selected_league != "Tutte":
    filtered_df = filtered_df[filtered_df["league"] == selected_league]

if selected_anno != "Tutti":
    filtered_df = filtered_df[filtered_df["anno"] == selected_anno]

if selected_home != "Tutte":
    filtered_df = filtered_df[filtered_df["home_team"] == selected_home]

if selected_away != "Tutte":
    filtered_df = filtered_df[filtered_df["away_team"] == selected_away]

# --- Funzione Calcolo ROI ---
def calcola_roi(matches_df, segno):
    if matches_df.empty:
        return {"Matches": 0, "Win%": 0, "BackPts": 0, "LayPts": 0, "ROI_Back%": 0, "ROI_Lay%": 0, "Odd Minima": "-"}

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
    roi_back = round(back_pts / matches * 100, 2) if matches > 0 else 0
    roi_lay = round(lay_pts / matches * 100, 2) if matches > 0 else 0
    odd_minima = round(100 / winrate, 2) if winrate > 0 else "-"

    return {"Matches": matches, "Win%": winrate, "BackPts": back_pts, "LayPts": lay_pts, "ROI_Back%": roi_back, "ROI_Lay%": roi_lay, "Odd Minima": odd_minima}

# --- Tabella ROI ---
roi_results = []
for segno in ["HOME", "DRAW", "AWAY"]:
    roi_results.append([selected_label, segno] + list(calcola_roi(filtered_df, segno).values()))

results_df = pd.DataFrame(roi_results, columns=["Label", "Segno", "Matches", "Win %", "Back Pts", "Lay Pts", "ROI_Back%", "ROI_Lay%", "Odd Minima"])

def color_rois(val):
    if isinstance(val, (int, float)) and val > 0:
        return 'background-color: #b6e8b6'
    return ''

st.subheader(f"Risultati ROI per {selected_label}")
st.dataframe(results_df.style.applymap(color_rois, subset=["Back Pts", "Lay Pts", "ROI_Back%", "ROI_Lay%"]))

# --- Funzione Risultati Esatti ---
def mostra_risultati_esatti(df_input, col_risultato):
    if col_risultato not in df_input.columns:
        st.warning(f"Colonna {col_risultato} non presente.")
        return

    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]

    def classifica_risultato(ris):
        try:
            home, away = map(int, ris.split("-"))
        except:
            return "Altro"
        if ris in risultati_interessanti:
            return ris
        if home > away:
            return "Altro risultato casa vince"
        elif home < away:
            return "Altro risultato ospite vince"
        else:
            return "Altro pareggio"

    df_input = df_input[df_input[col_risultato].notna()].copy()
    df_input["classificato"] = df_input[col_risultato].apply(classifica_risultato)
    distribuzione = df_input["classificato"].value_counts().reset_index()
    distribuzione.columns = [col_risultato, "Conteggio"]
    distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df_input) * 100).round(2)
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

    st.subheader(f"Risultati Esatti {col_risultato} ({len(df_input)} partite)")
    st.table(distribuzione)

# --- Analisi Squadre Selezionate ---
if selected_home != "Tutte" or selected_away != "Tutte":
    squadre_matches = filtered_df.copy()
    if selected_home != "Tutte":
        squadre_matches = squadre_matches[squadre_matches["home_team"] == selected_home]
    if selected_away != "Tutte":
        squadre_matches = squadre_matches[squadre_matches["away_team"] == selected_away]

    st.subheader(f"Statistiche per {selected_home} - {selected_away} ({selected_label})")

    mostra_risultati_esatti(squadre_matches, "risultato_ht")
    mostra_risultati_esatti(squadre_matches, "risultato_ft")
