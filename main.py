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

# --- Funzione per colorare i valori positivi e negativi ---
def color_positive_negative(val):
    """Funzione per colorare i valori positivi e negativi."""
    if isinstance(val, (int, float)):
        color = 'green' if val > 0 else 'red'
        return f'color: {color}'
    return None
    
# --- Nuova funzione per lo stile del DataFrame con MultiIndex (non più usata, ma tenuta per riferimento) ---
def style_multiindex_df(df):
    """
    Genera un DataFrame di stili per colorare 'Pts' e 'Roi' di un MultiIndex DataFrame.
    NOTA: Questa funzione non viene più utilizzata per evitare l'errore di indicizzazione.
    """
    color_df = pd.DataFrame('', index=df.index, columns=df.columns)
    
    for col in df.columns:
        if isinstance(col, tuple) and col[-1] in ['Pts', 'Roi']:
            color_df[col] = df[col].apply(lambda x: 'color: green;' if x > 0 else 'color: red;')
            
    return color_df

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
        
        # Funzione di stile inline per la sezione generale
        def color_positive_negative(val):
            if isinstance(val, (int, float)):
                color = 'green' if val > 0 else 'red'
                return f'color: {color}'
            return None
        
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

# Applica il filtro campionato come base
filtered_df_teams = df_original.copy()
if selected_campionato != 'Tutti' and 'league' in filtered_df_teams.columns:
    filtered_df_teams = filtered_df_teams[filtered_df_teams['league'] == selected_campionato]

# Determina quale filtro quote usare come principale
# Questo flag è il punto chiave della nuova logica.
# Se le quote away sono state modificate, usiamo quelle come filtro principale.
use_away_odd_filter = (min_odd_away != 1.01) or (max_odd_away != 50.0)

results_data_specific = []

# Analisi per la squadra di casa selezionata
if selected_home != 'Tutte':
    df_home_matches = filtered_df_teams[filtered_df_teams['home_team'] == selected_home].copy()
    
    if use_away_odd_filter:
        # Se il filtro principale è Quota Away, filtriamo le partite della squadra di casa
        # in base alla Quota Away dell'avversario.
        df_home_matches = df_home_matches[
            (df_home_matches['odd_away'] >= min_odd_away) &
            (df_home_matches['odd_away'] <= max_odd_away)
        ]
    else:
        # Se il filtro principale è Quota Home (default), filtriamo in base
        # alla Quota Home della squadra stessa.
        df_home_matches = df_home_matches[
            (df_home_matches['odd_home'] >= min_odd_home) &
            (df_home_matches['odd_home'] <= max_odd_home)
        ]

    # Calcolo dei ritorni su questo set di partite filtrate per tutti gli esiti
    if not df_home_matches.empty:
        home_back_results = calculate_returns(df_home_matches, 'home', 'Back')
        home_lay_results = calculate_returns(df_home_matches, 'home', 'Lay')
        draw_back_results = calculate_returns(df_home_matches, 'draw', 'Back')
        draw_lay_results = calculate_returns(df_home_matches, 'draw', 'Lay')
        away_back_results = calculate_returns(df_home_matches, 'away', 'Back')
        away_lay_results = calculate_returns(df_home_matches, 'away', 'Lay')

        row_data = {
            'Label': f"{selected_home} (Casa)",
            ('Matches', '', 'Total'): len(df_home_matches),
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

# Analisi per la squadra in trasferta selezionata
if selected_away != 'Tutte':
    df_away_matches = filtered_df_teams[filtered_df_teams['away_team'] == selected_away].copy()

    if use_away_odd_filter:
        # Se il filtro principale è Quota Away, filtriamo le partite della squadra in trasferta
        # in base alla sua stessa Quota Away.
        df_away_matches = df_away_matches[
            (df_away_matches['odd_away'] >= min_odd_away) &
            (df_away_matches['odd_away'] <= max_odd_away)
        ]
    else:
        # Se il filtro principale è Quota Home (default), filtriamo in base
        # alla Quota Home della squadra avversaria.
        df_away_matches = df_away_matches[
            (df_away_matches['odd_home'] >= min_odd_home) &
            (df_away_matches['odd_home'] <= max_odd_home)
        ]
    
    # Calcolo dei ritorni su questo set di partite filtrate per tutti gli esiti
    if not df_away_matches.empty:
        home_back_results = calculate_returns(df_away_matches, 'home', 'Back')
        home_lay_results = calculate_returns(df_away_matches, 'home', 'Lay')
        draw_back_results = calculate_returns(df_away_matches, 'draw', 'Back')
        draw_lay_results = calculate_returns(df_away_matches, 'draw', 'Lay')
        away_back_results = calculate_returns(df_away_matches, 'away', 'Back')
        away_lay_results = calculate_returns(df_away_matches, 'away', 'Lay')

        row_data = {
            'Label': f"{selected_away} (Trasferta)",
            ('Matches', '', 'Total'): len(df_away_matches),
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
    results_df = pd.DataFrame(results_data_specific).set_index('Label')
    
    # Appiattimento delle colonne del MultiIndex per evitare l'errore
    results_df.columns = results_df.columns.map(lambda col: ' - '.join(c for c in col if c != ''))

    # Funzione di stile per le colonne appiattite
    def style_flat_df(df):
        return pd.DataFrame(
            [['color: green;' if 'Pts' in c or 'Roi' in c and x > 0 else 'color: red;' if 'Pts' in c or 'Roi' in c and x < 0 else '' for c, x in r.items()] for _, r in df.iterrows()],
            index=df.index,
            columns=df.columns
        )

    # Applica lo stile utilizzando la nuova funzione
    styled_df = results_df.style.apply(style_flat_df, axis=None)
    
    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("Nessuna squadra da analizzare con i filtri selezionati.")
