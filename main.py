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

# --- Funzioni per il calcolo dei ritorni e la formattazione ---
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
    stake = 1.0
    if df.empty:
        return {
            'Win%': 0, 
            'Pts': 0, 
            'Roi': 0
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
                # Calcola il profitto Lay corretto
                total_profit += stake / (odd - 1) if odd > 1 else 0
            else:
                total_profit -= stake # Perdita Lay

    winrate = (wins / total_bets) * 100 if total_bets > 0 else 0
    roi = (total_profit / (total_bets * stake)) * 100 if total_bets > 0 else 0

    return {
        'Win%': round(winrate, 2),
        'Pts': round(total_profit, 2),
        'Roi': round(roi, 2)
    }

def color_positive_negative(val):
    """Funzione per colorare i valori positivi e negativi."""
    if isinstance(val, (int, float)):
        color = 'green' if val > 0 else 'red'
        return f'color: {color}'
    return None

# --- SIDEBAR: Filtri per l'analisi generale ---
st.sidebar.header("Filtri Partite")

# Filtro Campionato
if 'league' in df_original.columns:
    campionati = ['Tutti'] + sorted(df_original['league'].unique().tolist())
    selected_campionato = st.sidebar.selectbox('Seleziona Campionato', campionati)
else:
    st.sidebar.warning("Colonna 'league' non trovata.")
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
col_odds_1, col_odds_2 = st.sidebar.columns(2)
with col_odds_1:
    min_odd_home = st.number_input('Min Quota Home', value=1.01, step=0.01)
    min_odd_draw = st.number_input('Min Quota Pareggio', value=1.01, step=0.01)
    min_odd_away = st.number_input('Min Quota Away', value=1.01, step=0.01)
with col_odds_2:
    max_odd_home = st.number_input('Max Quota Home', value=50.0, step=0.01)
    max_odd_draw = st.number_input('Max Quota Pareggio', value=50.0, step=0.01)
    max_odd_away = st.number_input('Max Quota Away', value=50.0, step=0.01)

st.markdown("---")
# ==============================================================================
# --- SEZIONE 1: ANALISI GENERALE (IGNORA FILTRI SQUADRA) ---
# ==============================================================================
st.header("1. Analisi Generale")
st.write("Analisi dei ritorni per tutte le partite che soddisfano i filtri di Campionato e Quote.")

# Applicazione dei filtri per l'analisi generale (senza filtri squadra)
filtered_df_general = df_original.copy()
if selected_campionato != 'Tutti' and 'league' in filtered_df_general.columns:
    filtered_df_general = filtered_df_general[filtered_df_general['league'] == selected_campionato]
if 'odd_home' in filtered_df_general.columns:
    filtered_df_general = filtered_df_general[(filtered_df_general['odd_home'] >= min_odd_home) & (filtered_df_general['odd_home'] <= max_odd_home)]
if 'odd_draw' in filtered_df_general.columns:
    filtered_df_general = filtered_df_general[(filtered_df_general['odd_draw'] >= min_odd_draw) & (filtered_df_general['odd_draw'] <= max_odd_draw)]
if 'odd_away' in filtered_df_general.columns:
    filtered_df_general = filtered_df_general[(filtered_df_general['odd_away'] >= min_odd_away) & (filtered_df_general['odd_away'] <= max_odd_away)]

st.subheader(f"Partite filtrate: {len(filtered_df_general)} trovate")

if filtered_df_general.empty:
    st.warning("Nessuna partita trovata con i filtri di Campionato e Quote selezionati.")
else:
    results_data_general = []
    if 'odd_home' in filtered_df_general.columns:
        results_data_general.append({"Esito": "1 (Casa)", "Tipo Scommessa": "Back", 'Ritorno Punti': calculate_returns(filtered_df_general, 'home', 'Back')['Pts'], 'WinRate %': calculate_returns(filtered_df_general, 'home', 'Back')['Win%'], 'ROI %': calculate_returns(filtered_df_general, 'home', 'Back')['Roi']})
        results_data_general.append({"Esito": "1 (Casa)", "Tipo Scommessa": "Lay", 'Ritorno Punti': calculate_returns(filtered_df_general, 'home', 'Lay')['Pts'], 'WinRate %': calculate_returns(filtered_df_general, 'home', 'Lay')['Win%'], 'ROI %': calculate_returns(filtered_df_general, 'home', 'Lay')['Roi']})
    if 'odd_draw' in filtered_df_general.columns:
        results_data_general.append({"Esito": "X (Pareggio)", "Tipo Scommessa": "Back", 'Ritorno Punti': calculate_returns(filtered_df_general, 'draw', 'Back')['Pts'], 'WinRate %': calculate_returns(filtered_df_general, 'draw', 'Back')['Win%'], 'ROI %': calculate_returns(filtered_df_general, 'draw', 'Back')['Roi']})
        results_data_general.append({"Esito": "X (Pareggio)", "Tipo Scommessa": "Lay", 'Ritorno Punti': calculate_returns(filtered_df_general, 'draw', 'Lay')['Pts'], 'WinRate %': calculate_returns(filtered_df_general, 'draw', 'Lay')['Win%'], 'ROI %': calculate_returns(filtered_df_general, 'draw', 'Lay')['Roi']})
    if 'odd_away' in filtered_df_general.columns:
        results_data_general.append({"Esito": "2 (Trasferta)", "Tipo Scommessa": "Back", 'Ritorno Punti': calculate_returns(filtered_df_general, 'away', 'Back')['Pts'], 'WinRate %': calculate_returns(filtered_df_general, 'away', 'Back')['Win%'], 'ROI %': calculate_returns(filtered_df_general, 'away', 'Back')['Roi']})
        results_data_general.append({"Esito": "2 (Trasferta)", "Tipo Scommessa": "Lay", 'Ritorno Punti': calculate_returns(filtered_df_general, 'away', 'Lay')['Pts'], 'WinRate %': calculate_returns(filtered_df_general, 'away', 'Lay')['Win%'], 'ROI %': calculate_returns(filtered_df_general, 'away', 'Lay')['Roi']})

    if results_data_general:
        results_df_general = pd.DataFrame(results_data_general)
        st.dataframe(results_df_general.style.applymap(color_positive_negative, subset=['Ritorno Punti', 'ROI %']), use_container_width=True)
    else:
        st.warning("Impossibile calcolare i risultati. Verificare le colonne del database.")
        
with st.expander("Mostra Dati Filtrati", expanded=False):
    st.dataframe(filtered_df_general)

st.markdown("---")
# ==============================================================================
# --- SEZIONE 2: Analisi per Squadra (USA TUTTI I FILTRI) ---
# ==============================================================================
st.header("2. Analisi Ritorno per Squadra")
st.write("Rendimento di ogni singola squadra nel dataset filtrato.")

# Applicazione dei filtri di campionato e quote per l'analisi per squadra
filtered_df_teams = df_original.copy()
if selected_campionato != 'Tutti' and 'league' in filtered_df_teams.columns:
    filtered_df_teams = filtered_df_teams[filtered_df_teams['league'] == selected_campionato]
if 'odd_home' in filtered_df_teams.columns:
    filtered_df_teams = filtered_df_teams[(filtered_df_teams['odd_home'] >= min_odd_home) & (filtered_df_teams['odd_home'] <= max_odd_home)]
if 'odd_draw' in filtered_df_teams.columns:
    filtered_df_teams = filtered_df_teams[(filtered_df_teams['odd_draw'] >= min_odd_draw) & (filtered_df_teams['odd_draw'] <= max_odd_draw)]
if 'odd_away' in filtered_df_teams.columns:
    filtered_df_teams = filtered_df_teams[(filtered_df_teams['odd_away'] >= min_odd_away) & (filtered_df_teams['odd_away'] <= max_odd_away)]

# Applica i filtri delle squadre con la logica corretta (OR)
if selected_home != 'Tutte' and selected_away != 'Tutte':
    filtered_df_teams = filtered_df_teams[
        (filtered_df_teams['home_team'] == selected_home) | 
        (filtered_df_teams['away_team'] == selected_away)
    ]
elif selected_home != 'Tutte':
    filtered_df_teams = filtered_df_teams[filtered_df_teams['home_team'] == selected_home]
elif selected_away != 'Tutte':
    filtered_df_teams = filtered_df_teams[filtered_df_teams['away_team'] == selected_away]


if filtered_df_teams.empty:
    st.info("Nessuna squadra da analizzare con i filtri selezionati.")
else:
    # Ottieni tutte le squadre uniche presenti nel DataFrame filtrato
    all_teams_filtered = sorted(list(set(filtered_df_teams['home_team'].unique()).union(filtered_df_teams['away_team'].unique())))
    
    results_data_specific = []
    
    # Cicla su ogni squadra per calcolare i ritorni
    for team in all_teams_filtered:
        # Filtra le partite in cui la squadra gioca in casa
        df_home_matches = filtered_df_teams[filtered_df_teams['home_team'] == team]
        # Filtra le partite in cui la squadra gioca in trasferta
        df_away_matches = filtered_df_teams[filtered_df_teams['away_team'] == team]
        
        # Calcola il numero totale di partite per la squadra
        total_matches = len(df_home_matches) + len(df_away_matches)
        
        if total_matches > 0:
            # Calcola i ritorni per gli scenari Home, Draw, Away
            home_back_results = calculate_returns(df_home_matches, 'home', 'Back')
            home_lay_results = calculate_returns(df_home_matches, 'home', 'Lay')
            
            draw_back_results = calculate_returns(filtered_df_teams[(filtered_df_teams['home_team'] == team) | (filtered_df_teams['away_team'] == team)], 'draw', 'Back')
            draw_lay_results = calculate_returns(filtered_df_teams[(filtered_df_teams['home_team'] == team) | (filtered_df_teams['away_team'] == team)], 'draw', 'Lay')
            
            away_back_results = calculate_returns(df_away_matches, 'away', 'Back')
            away_lay_results = calculate_returns(df_away_matches, 'away', 'Lay')
            
            # Costruisci la riga per la tabella con tuple come chiavi per un MultiIndex uniforme
            row_data = {
                'Label': team,
                ('Matches', '', 'Total'): total_matches,
                ('Home', 'Back', 'Win%'): home_back_results['Win%'],
                ('Home', 'Back', 'Pts'): home_back_results['Pts'],
                ('Home', 'Back', 'Roi'): home_back_results['Roi'],
                ('Home', 'Lay', 'Win%'): home_lay_results['Win%'],
                ('Home', 'Lay', 'Pts'): home_lay_results['Pts'],
                ('Home', 'Lay', 'Roi'): home_lay_results['Roi'],
                
                ('Draw', 'Back', 'Win%'): draw_back_results['Win%'],
                ('Draw', 'Back', 'Pts'): draw_back_results['Pts'],
                ('Draw', 'Back', 'Roi'): draw_back_results['Roi'],
                ('Draw', 'Lay', 'Win%'): draw_lay_results['Win%'],
                ('Draw', 'Lay', 'Pts'): draw_lay_results['Pts'],
                ('Draw', 'Lay', 'Roi'): draw_lay_results['Roi'],
                
                ('Away', 'Back', 'Win%'): away_back_results['Win%'],
                ('Away', 'Back', 'Pts'): away_back_results['Pts'],
                ('Away', 'Back', 'Roi'): away_back_results['Roi'],
                ('Away', 'Lay', 'Win%'): away_lay_results['Win%'],
                ('Away', 'Lay', 'Pts'): away_lay_results['Pts'],
                ('Away', 'Lay', 'Roi'): away_lay_results['Roi']
            }
            results_data_specific.append(row_data)

    if results_data_specific:
        # Creazione del DataFrame con MultiIndex automatico dalle chiavi a tuple
        results_df = pd.DataFrame(results_data_specific).set_index('Label')
        
        # Creazione di una lista di tuple per le colonne da formattare
        # Questo approccio evita l'errore di indicizzazione
        cols_to_style = [col for col in results_df.columns if col[2] in ['Pts', 'Roi']]
        
        # Applica la formattazione e visualizza la tabella
        styled_df = results_df.style.applymap(color_positive_negative, subset=cols_to_style)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("Nessuna squadra da analizzare con i filtri selezionati.")
