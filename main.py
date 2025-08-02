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
campionati = ['Tutti'] + sorted(df_original['campionato'].unique().tolist())
selected_campionato = st.sidebar.selectbox('Seleziona Campionato', campionati)

# Filtro Squadra Home
squadre_home = ['Tutte'] + sorted(df_original['home_team'].unique().tolist())
selected_home = st.sidebar.selectbox('Seleziona Squadra Home', squadre_home)

# Filtro Squadra Away
squadre_away = ['Tutte'] + sorted(df_original['away_team'].unique().tolist())
selected_away = st.sidebar.selectbox('Seleziona Squadra Away', squadre_away)

# Filtro Quote
st.sidebar.subheader("Fasce di Quote")

# Recupera i valori min e max per le quote, gestendo eventuali KeyError
try:
    min_odd_home = df_original['odd_home'].min()
    max_odd_home = df_original['odd_home'].max()
    odd_home_range = st.sidebar.slider(
        'Quota Home',
        float(min_odd_home), float(max_odd_home), (float(min_odd_home), float(max_odd_home)), step=0.01
    )

    min_odd_draw = df_original['odd_draw'].min()
    max_odd_draw = df_original['odd_draw'].max()
    odd_draw_range = st.sidebar.slider(
        'Quota Pareggio',
        float(min_odd_draw), float(max_odd_draw), (float(min_odd_draw), float(max_odd_draw)), step=0.01
    )

    min_odd_away = df_original['odd_away'].min()
    max_odd_away = df_original['odd_away'].max()
    odd_away_range = st.sidebar.slider(
        'Quota Away',
        float(min_odd_away), float(max_odd_away), (float(min_odd_away), float(max_odd_away)), step=0.01
    )
except KeyError:
    st.sidebar.warning("Non tutte le colonne per le quote sono disponibili. Verificare che siano presenti 'odd_home', 'odd_draw', 'odd_away'.")
    odd_home_range = (0.0, 100.0)
    odd_draw_range = (0.0, 100.0)
    odd_away_range = (0.0, 100.0)
    
# --- Applicazione dei filtri ---
filtered_df = df_original.copy()

if selected_campionato != 'Tutti':
    filtered_df = filtered_df[filtered_df['campionato'] == selected_campionato]
if selected_home != 'Tutte':
    filtered_df = filtered_df[filtered_df['home_team'] == selected_home]
if selected_away != 'Tutte':
    filtered_df = filtered_df[filtered_df['away_team'] == selected_away]

# Filtro per le quote, gestendo l'assenza delle colonne
if 'odd_home' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['odd_home'] >= odd_home_range[0]) & 
        (filtered_df['odd_home'] <= odd_home_range[1])
    ]
if 'odd_draw' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['odd_draw'] >= odd_draw_range[0]) & 
        (filtered_df['odd_draw'] <= odd_draw_range[1])
    ]
if 'odd_away' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['odd_away'] >= odd_away_range[0]) & 
        (filtered_df['odd_away'] <= odd_away_range[1])
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

def calculate_returns(df, outcome_type):
    """Calcola i risultati per un dato esito (home, draw, away)."""
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
    
    # Assumiamo che le colonne esistano dopo la gestione dell'errore iniziale
    odd_col_name = f'odd_{outcome_type}'
    
    for _, row in df.iterrows():
        total_bets += 1
        result = get_result(row)
        odd = row[odd_col_name]
        
        # Calcolo per BACK (puntare)
        if st.session_state.bet_type == 'Back':
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

# Selezione tipo di scommessa
st.session_state.bet_type = st.radio(
    "Scegli il tipo di scommessa:",
    ('Back', 'Lay'),
    index=0,
    format_func=lambda x: "Back (Puntare)" if x == "Back" else "Lay (Bancare)"
)

# Calcola e mostra i risultati per ogni esito
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Esito: Casa Vince (1)")
    if 'odd_home' in filtered_df.columns:
        results_home = calculate_returns(filtered_df, 'home')
        st.metric(label="Ritorno Punti", value=results_home['Ritorno Punti'])
        st.metric(label="WinRate", value=f"{results_home['WinRate %']}%")
        st.metric(label="ROI", value=f"{results_home['ROI %']}%")
        st.metric(label="Odd Minima", value=results_home['Odd Minima'])
    else:
        st.warning("Colonna 'odd_home' non trovata.")

with col2:
    st.subheader("Esito: Pareggio (X)")
    if 'odd_draw' in filtered_df.columns:
        results_draw = calculate_returns(filtered_df, 'draw')
        st.metric(label="Ritorno Punti", value=results_draw['Ritorno Punti'])
        st.metric(label="WinRate", value=f"{results_draw['WinRate %']}%")
        st.metric(label="ROI", value=f"{results_draw['ROI %']}%")
        st.metric(label="Odd Minima", value=results_draw['Odd Minima'])
    else:
        st.warning("Colonna 'odd_draw' non trovata.")

with col3:
    st.subheader("Esito: Trasferta Vince (2)")
    if 'odd_away' in filtered_df.columns:
        results_away = calculate_returns(filtered_df, 'away')
        st.metric(label="Ritorno Punti", value=results_away['Ritorno Punti'])
        st.metric(label="WinRate", value=f"{results_away['WinRate %']}%")
        st.metric(label="ROI", value=f"{results_away['ROI %']}%")
        st.metric(label="Odd Minima", value=results_away['Odd Minima'])
    else:
        st.warning("Colonna 'odd_away' non trovata.")

# --- Dati Filtrati (opzionale) ---
with st.expander("Mostra Dati Filtrati", expanded=False):
    st.dataframe(filtered_df)

