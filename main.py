import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Analizzatore Ritorno Scommesse (Back & Lay)")

# Funzione per connettersi al database e recuperare i dati
# Utilizza le st.secrets per la connessione sicura
@st.cache_data
def run_query():
    """Connette al database PostgreSQL e carica tutti i dati dalla tabella allcamp."""
    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        dbname=st.secrets["postgres"]["dbname"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        sslmode="require"
    )
    df = pd.read_sql('SELECT * FROM "allcamp";', conn)
    conn.close()
    
    # Pulizia dati: converte le quote in numeri, gestendo la virgola
    for col in ['odd_home', 'odd_draw', 'odd_away']:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '.'),
                errors='coerce'
            )
            
    # Aggiunge una colonna 'risultato_ft' per comodità
    if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
        df['risultato_ft'] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)
        
    return df

# --- Caricamento dati ---
try:
    with st.spinner('Caricamento dati in corso...'):
        df_original = run_query()
    st.success("Dati caricati con successo!")
    st.write(f"Numero totale di partite nel database: **{len(df_original)}**")
except Exception as e:
    st.error(f"Errore nel caricamento dei dati: {e}")
    st.stop()
    
# --- Sidebar per i filtri ---
st.sidebar.header("Filtri Partite")

# Filtro Campionato
if 'league' in df_original.columns:
    campionati = ['Tutti'] + sorted(df_original['league'].unique().tolist())
    selected_campionato = st.sidebar.selectbox('Seleziona Campionato', campionati)
else:
    st.sidebar.warning("Colonna 'league' non trovata. Usare 'campionato' o un altro nome.")
    selected_campionato = 'Tutti'

# Filtro Squadra Home
if 'home_team' in df_original.columns:
    squadre_home = ['Tutte'] + sorted(df_original['home_team'].unique().tolist())
    selected_home = st.sidebar.selectbox('Seleziona Squadra Home', squadre_home)
else:
    st.sidebar.warning("Colonna 'home_team' non trovata.")
    selected_home = 'Tutte'

# Filtro Squadra Away
if 'away_team' in df_original.columns:
    squadre_away = ['Tutte'] + sorted(df_original['away_team'].unique().tolist())
    selected_away = st.sidebar.selectbox('Seleziona Squadra Away', squadre_away)
else:
    st.sidebar.warning("Colonna 'away_team' non trovata.")
    selected_away = 'Tutte'

# Filtro Quote
st.sidebar.subheader("Fasce di Quote")

# Input manuale per le quote
col_odds_1, col_odds_2 = st.sidebar.columns(2)
with col_odds_1:
    min_odd_home = st.number_input('Min Quota Home', value=1.01, step=0.01)
    min_odd_draw = st.number_input('Min Quota Pareggio', value=1.01, step=0.01)
    min_odd_away = st.number_input('Min Quota Away', value=1.01, step=0.01)
with col_odds_2:
    max_odd_home = st.number_input('Max Quota Home', value=50.0, step=0.01)
    max_odd_draw = st.number_input('Max Quota Pareggio', value=50.0, step=0.01)
    max_odd_away = st.number_input('Max Quota Away', value=50.0, step=0.01)

# --- Applicazione dei filtri ---
filtered_df = df_original.copy()

if selected_campionato != 'Tutti' and 'league' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['league'] == selected_campionato]
if selected_home != 'Tutte' and 'home_team' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['home_team'] == selected_home]
if selected_away != 'Tutte' and 'away_team' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['away_team'] == selected_away]

# Filtro per le quote
if 'odd_home' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['odd_home'] >= min_odd_home) & 
        (filtered_df['odd_home'] <= max_odd_home)
    ]
if 'odd_draw' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['odd_draw'] >= min_odd_draw) & 
        (filtered_df['odd_draw'] <= max_odd_draw)
    ]
if 'odd_away' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['odd_away'] >= min_odd_away) & 
        (filtered_df['odd_away'] <= max_odd_away)
    ]

st.subheader(f"Partite filtrate: {len(filtered_df)} trovate")

if filtered_df.empty:
    st.warning("Nessuna partita trovata con i filtri selezionati.")
    st.stop()

# --- Calcoli e Analisi Back & Lay ---
st.header("Analisi Ritorno Economico")
stake = 1.0 # 1 punto = 1% dello stake totale

def get_result(row):
    """Determina il risultato della partita."""
    if row['gol_home_ft'] > row['gol_away_ft']:
        return 'home'
    elif row['gol_home_ft'] < row['gol_away_ft']:
        return 'away'
    else:
        return 'draw'

def calculate_returns(df, outcome_type, bet_type):
    """Calcola i risultati per un dato esito e tipo di scommessa."""
    if df.empty:
        return {
            'Ritorno Punti': 0, 
            'WinRate %': 0, 
            'ROI %': 0, 
            'Odd Minima': '-'
        }
        
    wins = 0
    total_profit = 0
    total_bets = 0
    
    odd_col_name = f'odd_{outcome_type}'
    
    for _, row in df.iterrows():
        total_bets += 1
        result = get_result(row)
        odd = row[odd_col_name]
        
        # Calcolo per BACK (puntare)
        if bet_type == 'Back':
            if result == outcome_type:
                wins += 1
                total_profit += (odd - 1) * stake
            else:
                total_profit -= stake
        # Calcolo per LAY (bancare)
        else: # 'Lay'
            if result != outcome_type:
                wins += 1
                total_profit += (stake / (odd - 1)) if odd > 1 else 0  # Profitto Lay
            else:
                total_profit -= stake # Perdita Lay

    winrate = (wins / total_bets) * 100 if total_bets > 0 else 0
    roi = (total_profit / (total_bets * stake)) * 100 if total_bets > 0 else 0
    odd_minima = round(100 / winrate, 2) if winrate > 0 else '-'

    return {
        'Ritorno Punti': round(total_profit, 2),
        'WinRate %': round(winrate, 2),
        'ROI %': round(roi, 2),
        'Odd Minima': odd_minima
    }

# Calcola e mostra i risultati per ogni esito
col1, col2, col3, col4, col5, col6 = st.columns(6)

if 'odd_home' in filtered_df.columns:
    with col1:
        st.subheader("1 (Back)")
        results = calculate_returns(filtered_df, 'home', 'Back')
        st.metric("Ritorno Punti", results['Ritorno Punti'])
        st.metric("WinRate", f"{results['WinRate %']}%")
        st.metric("ROI", f"{results['ROI %']}%")
        st.metric("Odd Minima", results['Odd Minima'])
    with col2:
        st.subheader("1 (Lay)")
        results = calculate_returns(filtered_df, 'home', 'Lay')
        st.metric("Ritorno Punti", results['Ritorno Punti'])
        st.metric("WinRate", f"{results['WinRate %']}%")
        st.metric("ROI", f"{results['ROI %']}%")
        st.metric("Odd Minima", results['Odd Minima'])
else:
    with col1:
        st.warning("Colonna 'odd_home' non trovata.")

if 'odd_draw' in filtered_df.columns:
    with col3:
        st.subheader("X (Back)")
        results = calculate_returns(filtered_df, 'draw', 'Back')
        st.metric("Ritorno Punti", results['Ritorno Punti'])
        st.metric("WinRate", f"{results['WinRate %']}%")
        st.metric("ROI", f"{results['ROI %']}%")
        st.metric("Odd Minima", results['Odd Minima'])
    with col4:
        st.subheader("X (Lay)")
        results = calculate_returns(filtered_df, 'draw', 'Lay')
        st.metric("Ritorno Punti", results['Ritorno Punti'])
        st.metric("WinRate", f"{results['WinRate %']}%")
        st.metric("ROI", f"{results['ROI %']}%")
        st.metric("Odd Minima", results['Odd Minima'])
else:
    with col3:
        st.warning("Colonna 'odd_draw' non trovata.")

if 'odd_away' in filtered_df.columns:
    with col5:
        st.subheader("2 (Back)")
        results = calculate_returns(filtered_df, 'away', 'Back')
        st.metric("Ritorno Punti", results['Ritorno Punti'])
        st.metric("WinRate", f"{results['WinRate %']}%")
        st.metric("ROI", f"{results['ROI %']}%")
        st.metric("Odd Minima", results['Odd Minima'])
    with col6:
        st.subheader("2 (Lay)")
        results = calculate_returns(filtered_df, 'away', 'Lay')
        st.metric("Ritorno Punti", results['Ritorno Punti'])
        st.metric("WinRate", f"{results['WinRate %']}%")
        st.metric("ROI", f"{results['ROI %']}%")
        st.metric("Odd Minima", results['Odd Minima'])
else:
    with col5:
        st.warning("Colonna 'odd_away' non trovata.")

# --- Dati Filtrati (opzionale) ---
with st.expander("Mostra Dati Filtrati", expanded=False):
    st.dataframe(filtered_df)
