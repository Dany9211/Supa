import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisi ROI & Statistiche", layout="wide")
st.title("Analisi ROI & Statistiche per Label Odds e Squadre")

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

# --- APPLICA FILTRI ---
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["label_odds"] == selected_label]

if selected_league != "Tutte":
    filtered_df = filtered_df[filtered_df["league"] == selected_league]
if selected_anno != "Tutti":
    filtered_df = filtered_df[filtered_df["anno"] == selected_anno]

# --- Funzione Calcolo ROI ---
def calcola_roi(matches_df, segno):
    if matches_df.empty:
        return {"Matches": 0, "Win%": 0, "BackPts": 0, "LayPts": 0, "ROI%": 0, "Odd Minima": "-"}

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

        risultati.at[i, "back_profit"] = (odd - 1) if vinta else -1
        risultati.at[i, "lay_profit"] = - (odd - 1) if vinta else 1

    matches = len(risultati)
    winrate = round((risultati["back_profit"] > 0).mean() * 100, 2)
    back_pts = round(risultati["back_profit"].sum(), 2)
    lay_pts = round(risultati["lay_profit"].sum(), 2)
    roi = round((back_pts / matches) * 100, 2) if matches > 0 else 0
    odd_min = round(100 / winrate, 2) if winrate > 0 else "-"
    return {"Matches": matches, "Win%": winrate, "BackPts": back_pts, "LayPts": lay_pts, "ROI%": roi, "Odd Minima": odd_min}

# --- Tabella ROI ---
def genera_tabella_roi(df_input, label_name):
    table = []
    for segno in ["HOME", "DRAW", "AWAY"]:
        table.append([label_name, segno] + list(calcola_roi(df_input, segno).values()))
    return table

results_table = genera_tabella_roi(filtered_df, selected_label)

# --- Filtra squadre per label ---
def filtra_squadre(df_input, home_team, away_team):
    if selected_label.startswith("Home"):
        home_df = df_input[df_input["home_team"] == home_team]
        away_df = df_input[df_input["away_team"] == away_team]
    elif selected_label.startswith("Away"):
        home_df = df_input[df_input["away_team"] == home_team]
        away_df = df_input[df_input["home_team"] == away_team]
    else:  # Supercompetitive
        home_df = df_input[(df_input["home_team"] == home_team) | (df_input["away_team"] == home_team)]
        away_df = df_input[(df_input["home_team"] == away_team) | (df_input["away_team"] == away_team)]
    return pd.concat([home_df, away_df])

squadre_matches = filtra_squadre(filtered_df, selected_home, selected_away)
if not squadre_matches.empty:
    squadre_results = genera_tabella_roi(squadre_matches, f"{selected_home} & {selected_away}")
    results_table.extend(squadre_results)

results_df = pd.DataFrame(results_table, columns=["LABEL", "SEGNO", "Matches", "Win%", "Back Pts", "Lay Pts", "ROI%", "Odd Minima"])

# --- Colorazione ROI ---
def color_rois(val):
    if isinstance(val, (int, float)) and val > 0:
        return 'background-color: #b6e8b6'
    return ''

st.subheader(f"Prospetto ROI per {selected_label}")
st.dataframe(results_df.style.applymap(color_rois, subset=["Back Pts", "Lay Pts", "ROI%"]))

# --- Risultati Esatti ---
def mostra_risultati_esatti(df, col_risultato, titolo):
    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]
    df_valid = df[df[col_risultato].notna() & (df[col_risultato].str.contains("-"))].copy()

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

    df_valid["classificato"] = df_valid[col_risultato].apply(classifica_risultato)
    distribuzione = df_valid["classificato"].value_counts().reset_index()
    distribuzione.columns = [titolo, "Conteggio"]
    distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df_valid) * 100).round(2)
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    return distribuzione

if not squadre_matches.empty:
    st.subheader(f"Statistiche per {selected_home} & {selected_away} ({len(squadre_matches)} partite)")

    # Risultati esatti
    st.write("**Risultati Esatti HT:**")
    st.table(mostra_risultati_esatti(squadre_matches, "risultato_ht", "HT"))
    st.write("**Risultati Esatti FT:**")
    st.table(mostra_risultati_esatti(squadre_matches, "risultato_ft", "FT"))

    # Over/Under & BTTS
    temp_ht = squadre_matches["risultato_ht"].str.split("-", expand=True).apply(pd.to_numeric, errors="coerce").fillna(0)
    squadre_matches["tot_goals_ht"] = temp_ht[0] + temp_ht[1]
    temp_ft = squadre_matches["risultato_ft"].str.split("-", expand=True).apply(pd.to_numeric, errors="coerce").fillna(0)
    squadre_matches["tot_goals_ft"] = temp_ft[0] + temp_ft[1]

    st.write("**Over Goals HT:**")
    over_ht = [[f"Over {t} HT", (squadre_matches["tot_goals_ht"] > t).sum(),
                round((squadre_matches["tot_goals_ht"] > t).mean() * 100, 2),
                round(100 / ((squadre_matches["tot_goals_ht"] > t).mean() * 100), 2) if (squadre_matches["tot_goals_ht"] > t).mean() > 0 else "-"]
               for t in [0.5, 1.5, 2.5]]
    st.table(pd.DataFrame(over_ht, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

    st.write("**Over Goals FT:**")
    over_ft = [[f"Over {t} FT", (squadre_matches["tot_goals_ft"] > t).sum(),
                round((squadre_matches["tot_goals_ft"] > t).mean() * 100, 2),
                round(100 / ((squadre_matches["tot_goals_ft"] > t).mean() * 100), 2) if (squadre_matches["tot_goals_ft"] > t).mean() > 0 else "-"]
               for t in [0.5, 1.5, 2.5, 3.5, 4.5]]
    st.table(pd.DataFrame(over_ft, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

    btts = ((temp_ft[0] > 0) & (temp_ft[1] > 0)).sum()
    perc_btts = round(btts / len(squadre_matches) * 100, 2)
    st.write(f"**BTTS SI:** {btts} ({perc_btts}%) - Odd Minima: {round(100/perc_btts, 2) if perc_btts > 0 else '-'}")
