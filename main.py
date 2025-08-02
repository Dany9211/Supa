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
# --- SEZIONE 1: ANALISI GENERALE ---
# ==============================================================================
st.header("1. Analisi Generale")
st.write("Analisi dei ritorni per tutte le partite che soddisfano i filtri della sidebar.")

# Applicazione dei filtri
filtered_df = df_original.copy()
if selected_campionato != 'Tutti' and 'league' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['league'] == selected_campionato]
if selected_home != 'Tutte' and 'home_team' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['home_team'] == selected_home]
if selected_away != 'Tutte' and 'away_team' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['away_team'] == selected_away]
if 'odd_home' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['odd_home'] >= min_odd_home) & (filtered_df['odd_home'] <= max_odd_home)]
if 'odd_draw' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['odd_draw'] >= min_odd_draw) & (filtered_df['odd_draw'] <= max_odd_draw)]
if 'odd_away' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['odd_away'] >= min_odd_away) & (filtered_df['odd_away'] <= max_odd_away)]

st.subheader(f"Partite filtrate: {len(filtered_df)} trovate")

if filtered_df.empty:
    st.warning("Nessuna partita trovata con i filtri selezionati.")
else:
    results_data_general = []
    if 'odd_home' in filtered_df.columns:
        results_data_general.append({"Esito": "1 (Casa)", "Tipo Scommessa": "Back", **calculate_returns(filtered_df, 'home', 'Back')})
        results_data_general.append({"Esito": "1 (Casa)", "Tipo Scommessa": "Lay", **calculate_returns(filtered_df, 'home', 'Lay')})
    if 'odd_draw' in filtered_df.columns:
        results_data_general.append({"Esito": "X (Pareggio)", "Tipo Scommessa": "Back", **calculate_returns(filtered_df, 'draw', 'Back')})
        results_data_general.append({"Esito": "X (Pareggio)", "Tipo Scommessa": "Lay", **calculate_returns(filtered_df, 'draw', 'Lay')})
    if 'odd_away' in filtered_df.columns:
        results_data_general.append({"Esito": "2 (Trasferta)", "Tipo Scommessa": "Back", **calculate_returns(filtered_df, 'away', 'Back')})
        results_data_general.append({"Esito": "2 (Trasferta)", "Tipo Scommessa": "Lay", **calculate_returns(filtered_df, 'away', 'Lay')})

    if results_data_general:
        results_df_general = pd.DataFrame(results_data_general)
        st.dataframe(results_df_general.style.applymap(color_positive_negative, subset=['Ritorno Punti', 'ROI %']), use_container_width=True)
    else:
        st.warning("Impossibile calcolare i risultati. Verificare le colonne del database.")
        
with st.expander("Mostra Dati Filtrati", expanded=False):
    st.dataframe(filtered_df)

st.markdown("---")
# ==============================================================================
# --- SEZIONE 2: ANALISI PERSONALIZZATA PER SQUADRE ---
# ==============================================================================
st.header("2. Analisi Personalizzata per Squadre")
st.write("Analisi del rendimento di due squadre in base a specifiche fasce di quote e al ruolo (Home/Away).")

if 'home_team' in df_original.columns and 'away_team' in df_original.columns:
    teams_list = sorted(list(set(df_original['home_team'].unique()).union(df_original['away_team'].unique())))

    col_team1, col_team2 = st.columns(2)
    with col_team1:
        team_1 = st.selectbox('Seleziona Squadra 1', teams_list, index=0)
    with col_team2:
        team_2 = st.selectbox('Seleziona Squadra 2', teams_list, index=1)
        
    st.markdown("---")
    
    filter_type = st.radio(
        "Scegli il tipo di quota di riferimento per il filtro:",
        ("Quota Home", "Quota Away"),
        key="custom_filter_type"
    )

    special_filter = st.checkbox("Filtra entrambe le squadre con quote < 3.00", key="special_odd_filter")
    
    if special_filter:
        min_odd_custom = 1.01
        max_odd_custom = 2.99
        st.info("Filtro speciale attivo: le quote sono impostate tra 1.01 e 2.99.")
    else:
        col_custom_odds_1, col_custom_odds_2 = st.columns(2)
        with col_custom_odds_1:
            min_odd_custom = st.number_input('Min Quota', value=1.01, step=0.01, key="min_odd_custom")
        with col_custom_odds_2:
            max_odd_custom = st.number_input('Max Quota', value=50.0, step=0.01, key="max_odd_custom")

    results_data_specific = []
    
    # --- Calcolo dei risultati per Squadra 1 ---
    if not df_original.empty:
        df_team1_filtered = pd.DataFrame()
        if filter_type == "Quota Home":
            df_team1_filtered = df_original[
                (df_original['home_team'] == team_1) & 
                (df_original['odd_home'].between(min_odd_custom, max_odd_custom))
            ]
        else: # Quota Away
            df_team1_filtered = df_original[
                (df_original['away_team'] == team_1) & 
                (df_original['odd_away'].between(min_odd_custom, max_odd_custom))
            ]
        
        if not df_team1_filtered.empty:
            for bet_type in ['Back', 'Lay']:
                results_data_specific.append({
                    "Squadra": team_1, "Tipo Scommessa": bet_type, "Esito": "1 (Vittoria)",
                    **calculate_returns(df_team1_filtered, 'home' if filter_type == "Quota Home" else 'away', bet_type)
                })
                results_data_specific.append({
                    "Squadra": team_1, "Tipo Scommessa": bet_type, "Esito": "X (Pareggio)",
                    **calculate_returns(df_team1_filtered, 'draw', bet_type)
                })
                results_data_specific.append({
                    "Squadra": team_1, "Tipo Scommessa": bet_type, "Esito": "2 (Sconfitta)",
                    **calculate_returns(df_team1_filtered, 'away' if filter_type == "Quota Home" else 'home', bet_type)
                })

    # --- Calcolo dei risultati per Squadra 2 ---
    if not df_original.empty:
        df_team2_filtered = pd.DataFrame()
        if filter_type == "Quota Home":
            df_team2_filtered = df_original[
                (df_original['away_team'] == team_2) & 
                (df_original['odd_away'].between(min_odd_custom, max_odd_custom))
            ]
        else: # Quota Away
            df_team2_filtered = df_original[
                (df_original['home_team'] == team_2) & 
                (df_original['odd_home'].between(min_odd_custom, max_odd_custom))
            ]
        
        if not df_team2_filtered.empty:
            for bet_type in ['Back', 'Lay']:
                results_data_specific.append({
                    "Squadra": team_2, "Tipo Scommessa": bet_type, "Esito": "1 (Vittoria)",
                    **calculate_returns(df_team2_filtered, 'away' if filter_type == "Quota Home" else 'home', bet_type)
                })
                results_data_specific.append({
                    "Squadra": team_2, "Tipo Scommessa": bet_type, "Esito": "X (Pareggio)",
                    **calculate_returns(df_team2_filtered, 'draw', bet_type)
                })
                results_data_specific.append({
                    "Squadra": team_2, "Tipo Scommessa": bet_type, "Esito": "2 (Sconfitta)",
                    **calculate_returns(df_team2_filtered, 'home' if filter_type == "Quota Home" else 'away', bet_type)
                })
    
    if results_data_specific:
        results_specific_df = pd.DataFrame(results_data_specific)
        st.dataframe(results_specific_df.style.applymap(color_positive_negative, subset=['Ritorno Punti', 'ROI %']), use_container_width=True)
    else:
        st.info("Seleziona le squadre e una fascia di quote per visualizzare i risultati.")

else:
    st.warning("Le colonne 'home_team' e/o 'away_team' non sono disponibili per questa analisi.")
