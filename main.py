import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Funzione per calcolare le probabilità e le value bet
def calculate_probabilities_and_value_bets(df):
    
    # Rinominare le colonne per coerenza con il modello
    df = df.rename(columns={
        'League': 'league',
        'Home_Team': 'HomeTeam',
        'Away_Team': 'AwayTeam',
        'Gol_Home_FT': 'FTHG',
        'Gol_Away_FT': 'FTAG',
        'Odd_Home': 'PSH',
        'Odd_Draw': 'PSD',
        'Odd__Away': 'PSA',
        'Home_Pos_Tot': 'HomePos',
        'Away_Pos_Tot': 'AwayPos'
    })

    # Conversione delle colonne delle quote in numeri (sostituendo la virgola con il punto)
    for col in ['PSH', 'PSD', 'PSA']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)
    
    # Codifica dei risultati in numeri (0: vittoria casa, 1: pareggio, 2: vittoria trasferta)
    def result_to_numeric(row):
        if row['FTHG'] > row['FTAG']:
            return 0
        elif row['FTHG'] == row['FTAG']:
            return 1
        else:
            return 2
    df['Result'] = df.apply(result_to_numeric, axis=1)

    # Feature engineering
    df['HomeOddsProb'] = 1 / df['PSH']
    df['DrawOddsProb'] = 1 / df['PSD']
    df['AwayOddsProb'] = 1 / df['PSA']

    # Seleziona le features
    features = ['HomeOddsProb', 'DrawOddsProb', 'AwayOddsProb', 'HomePos', 'AwayPos']
    
    # Inizializza il dataframe dei risultati
    results_df = pd.DataFrame()

    # Itera su ogni campionato
    for league in df['league'].unique():
        st.subheader(f'Campionato: {league}')
        
        league_df = df[df['league'] == league].copy()
        
        # Elimina le righe con valori mancanti
        league_df.dropna(subset=features, inplace=True)
        if league_df.empty or len(league_df) < 10:
            st.warning(f'Dati insufficienti o non validi per il campionato {league}.')
            continue

        X = league_df[features]
        y = league_df['Result']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        model.fit(X_train_scaled, y_train)

        y_pred_proba = model.predict_proba(X_test_scaled)
        
        X_test_with_proba = X_test.copy()
        X_test_with_proba['PredProbHome'] = y_pred_proba[:, 0]
        X_test_with_proba['PredProbDraw'] = y_pred_proba[:, 1]
        X_test_with_proba['PredProbAway'] = y_pred_proba[:, 2]

        test_df_original = league_df.loc[X_test_with_proba.index]
        X_test_with_proba = pd.concat([test_df_original[['HomeTeam', 'AwayTeam', 'PSH', 'PSD', 'PSA', 'Result']], X_test_with_proba], axis=1)

        value_bets_home = X_test_with_proba[(X_test_with_proba['PredProbHome'] * X_test_with_proba['PSH'] > 1)]
        value_bets_draw = X_test_with_proba[(X_test_with_proba['PredProbDraw'] * X_test_with_proba['PSD'] > 1)]
        value_bets_away = X_test_with_proba[(X_test_with_proba['PredProbAway'] * X_test_with_proba['PSA'] > 1)]

        roi_home = 0
        if not value_bets_home.empty:
            capital_home = len(value_bets_home)
            winnings_home = value_bets_home[value_bets_home['Result'] == 0]['PSH'].sum()
            roi_home = (winnings_home - capital_home) / capital_home * 100 if capital_home > 0 else 0
        
        roi_draw = 0
        if not value_bets_draw.empty:
            capital_draw = len(value_bets_draw)
            winnings_draw = value_bets_draw[value_bets_draw['Result'] == 1]['PSD'].sum()
            roi_draw = (winnings_draw - capital_draw) / capital_draw * 100 if capital_draw > 0 else 0
        
        roi_away = 0
        if not value_bets_away.empty:
            capital_away = len(value_bets_away)
            winnings_away = value_bets_away[value_bets_away['Result'] == 2]['PSA'].sum()
            roi_away = (winnings_away - capital_away) / capital_away * 100 if capital_away > 0 else 0

        st.write(f"ROI Home: **{roi_home:.2f}%** ({len(value_bets_home)} scommesse)")
        st.write(f"ROI Draw: **{roi_draw:.2f}%** ({len(value_bets_draw)} scommesse)")
        st.write(f"ROI Away: **{roi_away:.2f}%** ({len(value_bets_away)} scommesse)")

        if not value_bets_home.empty:
            st.markdown("### Value Bets per la vittoria in casa")
            st.dataframe(value_bets_home[['HomeTeam', 'AwayTeam', 'PSH', 'PredProbHome']].rename(columns={'PSH': 'Quota', 'PredProbHome': 'Probabilità Prevista'}))
        
        if not value_bets_draw.empty:
            st.markdown("### Value Bets per il pareggio")
            st.dataframe(value_bets_draw[['HomeTeam', 'AwayTeam', 'PSD', 'PredProbDraw']].rename(columns={'PSD': 'Quota', 'PredProbDraw': 'Probabilità Prevista'}))

        if not value_bets_away.empty:
            st.markdown("### Value Bets per la vittoria in trasferta")
            st.dataframe(value_bets_away[['HomeTeam', 'AwayTeam', 'PSA', 'PredProbAway']].rename(columns={'PSA': 'Quota', 'PredProbAway': 'Probabilità Prevista'}))

        results_df = pd.concat([results_df, pd.DataFrame([{
            'Campionato': league,
            'ROI Home': roi_home,
            'ROI Draw': roi_draw,
            'ROI Away': roi_away
        }])], ignore_index=True)
    
    st.markdown("---")
    st.markdown("## Riepilogo dei Risultati per Campionato")
    st.dataframe(results_df)

# Interfaccia Streamlit
st.title('Modello Predittivo di Value Bet')

st.markdown("""
Questo modello analizza i dati storici delle partite di calcio per identificare potenziali **value bet**.
Il modello calcola le probabilità attese di un esito (vittoria casa, pareggio, vittoria trasferta)
e le confronta con le quote dei bookmaker. Una value bet si verifica quando la probabilità implicita della quota
è inferiore alla probabilità calcolata dal modello.
""")

st.markdown("---")
st.markdown("## Carica i Dati")
st.markdown("Carica un file CSV con le seguenti colonne: `League`, `Home_Team`, `Away_Team`, `Gol_Home_FT`, `Gol_Away_FT`, `Odd_Home`, `Odd_Draw`, `Odd__Away`, `Home_Pos_Tot`, `Away_Pos_Tot`.")

uploaded_file = st.file_uploader("Scegli un file CSV", type="csv")

if uploaded_file is not None:
    try:
        # Carica il file CSV, gestendo gli errori di formato
        df = pd.read_csv(uploaded_file, sep=',', on_bad_lines='skip', engine='python')
        st.write("Anteprima dei dati caricati:")
        st.dataframe(df.head())
        
        required_cols = ['League', 'Home_Team', 'Away_Team', 'Gol_Home_FT', 'Gol_Away_FT', 'Odd_Home', 'Odd_Draw', 'Odd__Away', 'Home_Pos_Tot', 'Away_Pos_Tot']
        if all(col in df.columns for col in required_cols):
            if st.button("Esegui l'analisi"):
                with st.spinner('Elaborazione in corso...'):
                    calculate_probabilities_and_value_bets(df)
                st.success('Analisi completata!')
        else:
            missing_cols = [col for col in required_cols if col not in df.columns]
            st.error(f"Il file CSV deve contenere le seguenti colonne. Mancano: {', '.join(missing_cols)}")
    except Exception as e:
        st.error(f"Errore durante la lettura del file: {e}. Controlla il formato del file o il delimitatore.")
