import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Funzione per calcolare le probabilità e le value bet
def calculate_probabilities_and_value_bets(df):
    
    # Calcolo della classifica
    df['HomePos'] = df.groupby('league')['HomeTeam'].rank(method='dense')
    df['AwayPos'] = df.groupby('league')['AwayTeam'].rank(method='dense')
    
    # Codifica dei risultati in numeri
    def result_to_numeric(row):
        if row['FTHG'] > row['FTAG']:
            return 0  # Home Win
        elif row['FTHG'] == row['FTAG']:
            return 1  # Draw
        else:
            return 2  # Away Win
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
        
        # Filtra i dati per il campionato corrente
        league_df = df[df['league'] == league].copy()
        
        # Elimina le righe con valori mancanti nelle feature
        league_df.dropna(subset=features, inplace=True)
        if league_df.empty:
            st.warning(f'Nessun dato valido per il campionato {league}.')
            continue

        X = league_df[features]
        y = league_df['Result']

        # Divisione in training e testing set (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalizzazione delle features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Addestramento del modello
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # Previsioni e probabilità
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Calcolo dell'accuratezza
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuratezza del modello per il test set: {accuracy:.2%}")
        
        # Aggiungi le probabilità previste al test set
        X_test_with_proba = X_test.copy()
        X_test_with_proba['PredProbHome'] = y_pred_proba[:, 0]
        X_test_with_proba['PredProbDraw'] = y_pred_proba[:, 1]
        X_test_with_proba['PredProbAway'] = y_pred_proba[:, 2]

        # Aggiungi le colonne originali
        test_df_original = league_df.loc[X_test_with_proba.index]
        X_test_with_proba = pd.concat([test_df_original[['HomeTeam', 'AwayTeam', 'PSH', 'PSD', 'PSA', 'Result']], X_test_with_proba], axis=1)

        # Calcolo delle Value Bets
        value_bets_home = X_test_with_proba[(X_test_with_proba['PredProbHome'] * X_test_with_proba['PSH'] > 1)]
        value_bets_draw = X_test_with_proba[(X_test_with_proba['PredProbDraw'] * X_test_with_proba['PSD'] > 1)]
        value_bets_away = X_test_with_proba[(X_test_with_proba['PredProbAway'] * X_test_with_proba['PSA'] > 1)]

        # Calcolo del ROI
        # ROI Home
        roi_home = 0
        if not value_bets_home.empty:
            capital_home = len(value_bets_home)
            winnings_home = value_bets_home[value_bets_home['Result'] == 0]['PSH'].sum()
            roi_home = (winnings_home - capital_home) / capital_home * 100 if capital_home > 0 else 0
        
        # ROI Draw
        roi_draw = 0
        if not value_bets_draw.empty:
            capital_draw = len(value_bets_draw)
            winnings_draw = value_bets_draw[value_bets_draw['Result'] == 1]['PSD'].sum()
            roi_draw = (winnings_draw - capital_draw) / capital_draw * 100 if capital_draw > 0 else 0
        
        # ROI Away
        roi_away = 0
        if not value_bets_away.empty:
            capital_away = len(value_bets_away)
            winnings_away = value_bets_away[value_bets_away['Result'] == 2]['PSA'].sum()
            roi_away = (winnings_away - capital_away) / capital_away * 100 if capital_away > 0 else 0

        # Mostra i risultati
        st.write(f"ROI Home: **{roi_home:.2f}%** ({len(value_bets_home)} scommesse)")
        st.write(f"ROI Draw: **{roi_draw:.2f}%** ({len(value_bets_draw)} scommesse)")
        st.write(f"ROI Away: **{roi_away:.2f}%** ({len(value_bets_away)} scommesse)")

        # Mostra le scommesse identificate
        if not value_bets_home.empty:
            st.markdown("### Value Bets per la vittoria in casa")
            st.dataframe(value_bets_home[['HomeTeam', 'AwayTeam', 'PSH', 'PredProbHome']].rename(columns={'PSH': 'Quota', 'PredProbHome': 'Probabilità Prevista'}))
        
        if not value_bets_draw.empty:
            st.markdown("### Value Bets per il pareggio")
            st.dataframe(value_bets_draw[['HomeTeam', 'AwayTeam', 'PSD', 'PredProbDraw']].rename(columns={'PSD': 'Quota', 'PredProbDraw': 'Probabilità Prevista'}))

        if not value_bets_away.empty:
            st.markdown("### Value Bets per la vittoria in trasferta")
            st.dataframe(value_bets_away[['HomeTeam', 'AwayTeam', 'PSA', 'PredProbAway']].rename(columns={'PSA': 'Quota', 'PredProbAway': 'Probabilità Prevista'}))

        # Aggiungi i risultati al dataframe finale
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
st.markdown("Carica un file CSV con le seguenti colonne: `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `PSH`, `PSD`, `PSA`, `league`.")

uploaded_file = st.file_uploader("Scegli un file CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Anteprima dei dati caricati:")
    st.dataframe(df.head())
    
    if st.button("Esegui l'analisi"):
        with st.spinner('Elaborazione in corso...'):
            calculate_probabilities_and_value_bets(df)
        st.success('Analisi completata!')
