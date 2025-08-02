import streamlit as st
import pandas as pd
from supabase import create_client, Client
import numpy as np

# --- ISTRUZIONI DI INSTALLAZIONE ---
# Se riscontri un errore "ModuleNotFoundError", esegui questo comando nel tuo terminale:
# pip install streamlit supabase pandas
# --- FINE ISTRUZIONI ---

# --- CONFIGURAZIONE DEL DATABASE SUPABASE ---
# I dati di connessione devono essere salvati nel file .streamlit/secrets.toml
#
# Esempio di .streamlit/secrets.toml:
# [supabase]
# url = "https://<tuo_ID_progetto>.supabase.co"
# key = "<la_tua_chiave_anon>"

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Analisi Profittabilità Scommesse", layout="wide")
st.title("📊 Analisi Scommesse: Back vs Lay")
st.subheader("Filtra i dati e calcola le metriche di profitto su 1% di stake")

# --- CONNESSIONE E CARICAMENTO DATI DA SUPABASE ---
@st.cache_data
def init_supabase():
    """Crea un client Supabase con le credenziali da secrets.toml."""
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except KeyError as e:
        st.error(f"Errore: Credenziale Supabase mancante in secrets.toml: {e}")
        st.stop()

supabase = init_supabase()

@st.cache_data
def get_data_from_supabase():
    """Recupera tutti i dati dalla tabella 'matches' del database Supabase."""
    try:
        response = supabase.from_("matches").select("*").execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Errore durante il recupero dei dati da Supabase: {e}")
        return pd.DataFrame()

# --- CARICAMENTO E PULIZIA DATI ---
data_load_state = st.info("Caricamento dati in corso...")
df = get_data_from_supabase()
data_load_state.empty()

if df.empty:
    st.error("Nessun dato trovato. Controlla la connessione al database e il nome della tabella.")
    st.stop()

# Pulizia e conversione dei dati numerici
required_cols = ["league", "home_team", "away_team", "odd_home", "odd_draw", "odd_away", "gol_home_ft", "gol_away_ft"]
if not all(col in df.columns for col in required_cols):
    st.error("Una o più colonne necessarie non sono presenti nel database.")
    st.info(f"Colonne richieste: {required_cols}")
    st.stop()

df.dropna(subset=required_cols, inplace=True)
df.reset_index(drop=True, inplace=True)

for col in ["odd_home", "odd_draw", "odd_away"]:
    # Sostituisce la virgola con il punto e converte in numerico
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
for col in ["gol_home_ft", "gol_away_ft"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

# --- FUNZIONE DI CALCOLO METRICHE (BACK E LAY) ---
def calcola_metrica_scommessa(df_matches, bet_type, stake_percent=1):
    """
    Calcola le metriche di profitto (Winrate, Odd Breakeven, Ritorno Punti, ROI)
    per un determinato tipo di scommessa (Home, Draw, Away) e una stake fissa.
    """
    if df_matches.empty:
        return {
            "back_winrate": 0, "back_odd_breakeven": 0, "back_points": 0, "back_roi": 0,
            "lay_winrate": 0, "lay_odd_breakeven": 0, "lay_points": 0, "lay_roi": 0
        }

    total_bets = len(df_matches)
    # 1% di stake = 1 punto, quindi stake_percent è l'unità di misura.
    stake_unit = stake_percent
    
    odds_col = f"odd_{bet_type}"
    
    # Condizione di vittoria per il 'Back' (puntare)
    if bet_type == "home":
        back_win_condition = df_matches["gol_home_ft"] > df_matches["gol_away_ft"]
    elif bet_type == "draw":
        back_win_condition = df_matches["gol_home_ft"] == df_matches["gol_away_ft"]
    else: # away
        back_win_condition = df_matches["gol_home_ft"] < df_matches["gol_away_ft"]

    # Calcoli per il "Back"
    back_wins = back_win_condition.sum()
    back_winrate = (back_wins / total_bets) * 100 if total_bets > 0 else 0
    back_odd_breakeven = 100 / back_winrate if back_winrate > 0 else 0
    
    # Calcolo del ritorno in punti (Punti guadagnati se vinci, -1 se perdi)
    back_profits = np.where(back_win_condition, (df_matches[odds_col] - 1) * stake_unit, -stake_unit)
    back_points_return = back_profits.sum()
    back_roi = (back_points_return / (total_bets * stake_unit)) * 100 if total_bets > 0 else 0

    # Calcoli per il "Lay" (bancare)
    lay_win_condition = ~back_win_condition
    lay_wins = lay_win_condition.sum()
    lay_winrate = (lay_wins / total_bets) * 100 if total_bets > 0 else 0
    lay_odd_breakeven = 100 / lay_winrate if lay_winrate > 0 else 0
    
    # Calcolo del ritorno in punti (Punti guadagnati se vinci, -quota se perdi)
    lay_profits = np.where(lay_win_condition, stake_unit, - (df_matches[odds_col] - 1) * stake_unit)
    lay_points_return = lay_profits.sum()
    lay_roi = (lay_points_return / (total_bets * stake_unit)) * 100 if total_bets > 0 else 0
    
    return {
        "back_winrate": back_winrate, "back_odd_breakeven": back_odd_breakeven, 
        "back_points": back_points_return, "back_roi": back_roi,
        "lay_winrate": lay_winrate, "lay_odd_breakeven": lay_odd_breakeven, 
        "lay_points": lay_points_return, "lay_roi": lay_roi
    }

# --- FILTRI NELLA SIDEBAR ---
st.sidebar.header("Filtri")
st.sidebar.info(f"Partite totali nel database: {len(df)}")

# Filtri Campionato, Squadra Home, Squadra Away
leagues = sorted(df["league"].unique())
selected_leagues = st.sidebar.multiselect("Seleziona Campionato", leagues, default=leagues)

home_teams = sorted(df["home_team"].unique())
selected_home_teams = st.sidebar.multiselect("Seleziona Squadra Home", home_teams)

away_teams = sorted(df["away_team"].unique())
selected_away_teams = st.sidebar.multiselect("Seleziona Squadra Away", away_teams)

# Filtri quote con slider
st.sidebar.subheader("Fascia di Quote")
min_home_odd, max_home_odd = st.sidebar.slider("Quote Home", min_value=1.0, max_value=20.0, value=(1.0, 20.0), step=0.1)
min_draw_odd, max_draw_odd = st.sidebar.slider("Quote Pareggio", min_value=1.0, max_value=20.0, value=(1.0, 20.0), step=0.1)
min_away_odd, max_away_odd = st.sidebar.slider("Quote Away", min_value=1.0, max_value=20.0, value=(1.0, 20.0), step=0.1)

# --- APPLICAZIONE DEI FILTRI ---
filtered_df = df[
    (df["league"].isin(selected_leagues)) &
    (df["odd_home"] >= min_home_odd) & (df["odd_home"] <= max_home_odd) &
    (df["odd_draw"] >= min_draw_odd) & (df["odd_draw"] <= max_draw_odd) &
    (df["odd_away"] >= min_away_odd) & (df["odd_away"] <= max_away_odd)
]

if selected_home_teams:
    filtered_df = filtered_df[filtered_df["home_team"].isin(selected_home_teams)]
if selected_away_teams:
    filtered_df = filtered_df[filtered_df["away_team"].isin(selected_away_teams)]

# --- VISUALIZZAZIONE RISULTATI ---
st.subheader(f"Risultati per {len(filtered_df)} partite filtrate")

if filtered_df.empty:
    st.warning("Nessuna partita trovata con i filtri selezionati. Prova a modificare i filtri.")
    st.stop()

home_metrics = calcola_metrica_scommessa(filtered_df, 'home')
draw_metrics = calcola_metrica_scommessa(filtered_df, 'draw')
away_metrics = calcola_metrica_scommessa(filtered_df, 'away')

# Sezione "Puntare" (Back)
st.header("📈 Analisi 'Puntare' (Back)")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Home")
    st.metric("Winrate", f"{home_metrics['back_winrate']:.2f}%")
    st.metric("Odd minima (breakeven)", f"{home_metrics['back_odd_breakeven']:.2f}")
    st.metric("Ritorno (Points)", f"{home_metrics['back_points']:.2f}")
    st.metric("ROI", f"{home_metrics['back_roi']:.2f}%")
    
with col2:
    st.subheader("Pareggio")
    st.metric("Winrate", f"{draw_metrics['back_winrate']:.2f}%")
    st.metric("Odd minima (breakeven)", f"{draw_metrics['back_odd_breakeven']:.2f}")
    st.metric("Ritorno (Points)", f"{draw_metrics['back_points']:.2f}")
    st.metric("ROI", f"{draw_metrics['back_roi']:.2f}%")

with col3:
    st.subheader("Away")
    st.metric("Winrate", f"{away_metrics['back_winrate']:.2f}%")
    st.metric("Odd minima (breakeven)", f"{away_metrics['back_odd_breakeven']:.2f}")
    st.metric("Ritorno (Points)", f"{away_metrics['back_points']:.2f}")
    st.metric("ROI", f"{away_metrics['back_roi']:.2f}%")
    
st.markdown("---")

# Sezione "Bancare" (Lay)
st.header("📉 Analisi 'Bancare' (Lay)")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Lay Home")
    st.metric("Winrate", f"{home_metrics['lay_winrate']:.2f}%")
    st.metric("Odd minima (breakeven)", f"{home_metrics['lay_odd_breakeven']:.2f}")
    st.metric("Ritorno (Points)", f"{home_metrics['lay_points']:.2f}")
    st.metric("ROI", f"{home_metrics['lay_roi']:.2f}%")

with col2:
    st.subheader("Lay Pareggio")
    st.metric("Winrate", f"{draw_metrics['lay_winrate']:.2f}%")
    st.metric("Odd minima (breakeven)", f"{draw_metrics['lay_odd_breakeven']:.2f}")
    st.metric("Ritorno (Points)", f"{draw_metrics['lay_points']:.2f}")
    st.metric("ROI", f"{draw_metrics['lay_roi']:.2f}%")

with col3:
    st.subheader("Lay Away")
    st.metric("Winrate", f"{away_metrics['lay_winrate']:.2f}%")
    st.metric("Odd minima (breakeven)", f"{away_metrics['lay_odd_breakeven']:.2f}")
    st.metric("Ritorno (Points)", f"{away_metrics['lay_points']:.2f}")
    st.metric("ROI", f"{away_metrics['lay_roi']:.2f}%")

st.markdown("---")
if st.checkbox("Mostra partite filtrate"):
    st.dataframe(filtered_df)

