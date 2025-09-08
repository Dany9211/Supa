import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Funzione principale per l'analisi del modello (parte 1)
def calculate_probabilities_and_value_bets(df):
    
    # Rinominare le colonne per coerenza
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

    # Conversione delle colonne delle quote in numeri
    for col in ['PSH', 'PSD', 'PSA']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(0, np.nan)
    
    # Codifica dei risultati in numeri
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
    
    features = ['HomeOddsProb', 'DrawOddsProb', 'AwayOddsProb', 'HomePos', 'AwayPos']
    results_df = pd.DataFrame()

    for league in df['league'].unique():
        st.subheader(f'Campionato: {league}')
        
        league_df = df[df['league'] == league].copy()
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

        # Calcolo Value Bets (BACK)
        st.markdown("### Analisi Scommesse BACK (A Favore)")
        value_bets_home_back = X_test_with_proba[(X_test_with_proba['PredProbHome'] * X_test_with_proba['PSH'] > 1)]
        value_bets_draw_back = X_test_with_proba[(X_test_with_proba['PredProbDraw'] * X_test_with_proba['PSD'] > 1)]
        value_bets_away_back = X_test_with_proba[(X_test_with_proba['PredProbAway'] * X_test_with_proba['PSA'] > 1)]

        roi_home_back = (value_bets_home_back[value_bets_home_back['Result'] == 0]['PSH'].sum() - len(value_bets_home_back)) / len(value_bets_home_back) * 100 if len(value_bets_home_back) > 0 else 0
        roi_draw_back = (value_bets_draw_back[value_bets_draw_back['Result'] == 1]['PSD'].sum() - len(value_bets_draw_back)) / len(value_bets_draw_back) * 100 if len(value_bets_draw_back) > 0 else 0
        roi_away_back = (value_bets_away_back[value_bets_away_back['Result'] == 2]['PSA'].sum() - len(value_bets_away_back)) / len(value_bets_away_back) * 100 if len(value_bets_away_back) > 0 else 0

        st.write(f"ROI Home: **{roi_home_back:.2f}%** ({len(value_bets_home_back)} scommesse)")
        st.write(f"ROI Draw: **{roi_draw_back:.2f}%** ({len(value_bets_draw_back)} scommesse)")
        st.write(f"ROI Away: **{roi_away_back:.2f}%** ({len(value_bets_away_back)} scommesse)")

        # Calcolo Value Bets (LAY)
        st.markdown("### Analisi Scommesse LAY (Contro)")
        value_bets_home_lay = X_test_with_proba[((X_test_with_proba['PredProbDraw'] + X_test_with_proba['PredProbAway']) * (X_test_with_proba['PSH']/(X_test_with_proba['PSH']-1)) > 1)]
        value_bets_draw_lay = X_test_with_proba[((X_test_with_proba['PredProbHome'] + X_test_with_proba['PredProbAway']) * (X_test_with_proba['PSD']/(X_test_with_proba['PSD']-1)) > 1)]
        value_bets_away_lay = X_test_with_proba[((X_test_with_proba['PredProbHome'] + X_test_with_proba['PredProbDraw']) * (X_test_with_proba['PSA']/(X_test_with_proba['PSA']-1)) > 1)]
        
        roi_home_lay = (len(value_bets_home_lay[value_bets_home_lay['Result'] != 0]) - len(value_bets_home_lay[value_bets_home_lay['Result'] == 0]) * (value_bets_home_lay['PSH']-1).mean()) / len(value_bets_home_lay) * 100 if len(value_bets_home_lay) > 0 else 0
        roi_draw_lay = (len(value_bets_draw_lay[value_bets_draw_lay['Result'] != 1]) - len(value_bets_draw_lay[value_bets_draw_lay['Result'] == 1]) * (value_bets_draw_lay['PSD']-1).mean()) / len(value_bets_draw_lay) * 100 if len(value_bets_draw_lay) > 0 else 0
        roi_away_lay = (len(value_bets_away_lay[value_bets_away_lay['Result'] != 2]) - len(value_bets_away_lay[value_bets_away_lay['Result'] == 2]) * (value_bets_away_lay['PSA']-1).mean()) / len(value_bets_away_lay) * 100 if len(value_bets_away_lay) > 0 else 0

        st.write(f"ROI Lay Home: **{roi_home_lay:.2f}%** ({len(value_bets_home_lay)} scommesse)")
        st.write(f"ROI Lay Draw: **{roi_draw_lay:.2f}%** ({len(value_bets_draw_lay)} scommesse)")
        st.write(f"ROI Lay Away: **{roi_away_lay:.2f}%** ({len(value_bets_away_lay)} scommesse)")

        if not value_bets_home_back.empty:
            st.markdown("#### Scommesse BACK Home Win")
            st.dataframe(value_bets_home_back[['HomeTeam', 'AwayTeam', 'PSH', 'PredProbHome']].rename(columns={'PSH': 'Quota', 'PredProbHome': 'Probabilità Prevista'}))
        if not value_bets_draw_back.empty:
            st.markdown("#### Scommesse BACK Draw")
            st.dataframe(value_bets_draw_back[['HomeTeam', 'AwayTeam', 'PSD', 'PredProbDraw']].rename(columns={'PSD': 'Quota', 'PredProbDraw': 'Probabilità Prevista'}))
        if not value_bets_away_back.empty:
            st.markdown("#### Scommesse BACK Away Win")
            st.dataframe(value_bets_away_back[['HomeTeam', 'AwayTeam', 'PSA', 'PredProbAway']].rename(columns={'PSA': 'Quota', 'PredProbAway': 'Probabilità Prevista'}))
        
        results_df = pd.concat([results_df, pd.DataFrame([{
            'Campionato': league, 'ROI Home Back': roi_home_back, 'ROI Draw Back': roi_draw_back, 'ROI Away Back': roi_away_back,
            'ROI Home Lay': roi_home_lay, 'ROI Draw Lay': roi_draw_lay, 'ROI Away Lay': roi_away_lay
        }])], ignore_index=True)
    
    st.markdown("---")
    st.markdown("## Riepilogo dei Risultati per Campionato")
    st.dataframe(results_df)

# Funzione per prevedere una singola partita (parte 2)
def predict_single_match(model, scaler, home_pos, away_pos, odd_home, odd_draw, odd_away):
    
    # Prepara i dati per la previsione
    try:
        data = [[1/odd_home, 1/odd_draw, 1/odd_away, home_pos, away_pos]]
        new_match_df = pd.DataFrame(data, columns=['HomeOddsProb', 'DrawOddsProb', 'AwayOddsProb', 'HomePos', 'AwayPos'])
        
        new_match_scaled = scaler.transform(new_match_df)
        pred_proba = model.predict_proba(new_match_scaled)[0]
        
        return pred_proba
    except Exception as e:
        return None

# Interfaccia Streamlit
st.title('Modello Predittivo di Value Bet')

st.markdown("Questo strumento è diviso in due sezioni: una per l'analisi su dati storici e una per la previsione di una nuova partita singola.")

st.markdown("---")

st.header("1. Analisi su Dati Storici (BACK & LAY)")
st.markdown("Carica un file CSV per testare il modello su un dataset completo e valutare il ROI.")
uploaded_file = st.file_uploader("Scegli un file CSV", type="csv")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip', engine='python')
        st.write("Anteprima dei dati caricati:")
        st.dataframe(df.head())
        
        required_cols = ['League', 'Home_Team', 'Away_Team', 'Gol_Home_FT', 'Gol_Away_FT', 'Odd_Home', 'Odd_Draw', 'Odd__Away', 'Home_Pos_Tot', 'Away_Pos_Tot']
        if all(col in df.columns for col in required_cols):
            if st.button("Esegui Analisi Storica"):
                with st.spinner('Elaborazione in corso...'):
                    calculate_probabilities_and_value_bets(df)
                st.success('Analisi completata!')
        else:
            missing_cols = [col for col in required_cols if col not in df.columns]
            st.error(f"Il file CSV deve contenere le seguenti colonne. Mancano: {', '.join(missing_cols)}")
    except Exception as e:
        st.error(f"Errore durante la lettura del file: {e}")

st.markdown("---")

st.header("2. Previsione Nuova Partita")
st.markdown("Inserisci manualmente i dati per prevedere le probabilità di un match futuro. **Nota**: L'analisi di una singola partita funziona solo dopo aver eseguito l'analisi storica, perché il modello viene addestrato con i dati che hai caricato.")

if 'df_historical' not in st.session_state:
    st.session_state.df_historical = None

if uploaded_file is not None and st.button("Carica Dati per Previsione"):
    try:
        df_temp = pd.read_csv(uploaded_file, on_bad_lines='skip', engine='python')
        st.session_state.df_historical = df_temp
        st.success("Dati storici caricati con successo per la previsione!")
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati per la previsione: {e}")

if st.session_state.df_historical is not None:
    
    # Preparazione del modello per la previsione
    df_pred = st.session_state.df_historical.rename(columns={
        'League': 'league', 'Home_Team': 'HomeTeam', 'Away_Team': 'AwayTeam',
        'Gol_Home_FT': 'FTHG', 'Gol_Away_FT': 'FTAG', 'Odd_Home': 'PSH',
        'Odd_Draw': 'PSD', 'Odd__Away': 'PSA',
        'Home_Pos_Tot': 'HomePos', 'Away_Pos_Tot': 'AwayPos'
    })
    for col in ['PSH', 'PSD', 'PSA']:
        df_pred[col] = df_pred[col].astype(str).str.replace(',', '.', regex=False)
        df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')
        df_pred[col] = df_pred[col].replace(0, np.nan)
    df_pred['Result'] = df_pred.apply(lambda row: 0 if row['FTHG'] > row['FTAG'] else (1 if row['FTHG'] == row['FTAG'] else 2), axis=1)

    features = ['HomeOddsProb', 'DrawOddsProb', 'AwayOddsProb', 'HomePos', 'AwayPos']
    df_pred['HomeOddsProb'] = 1 / df_pred['PSH']
    df_pred['DrawOddsProb'] = 1 / df_pred['PSD']
    df_pred['AwayOddsProb'] = 1 / df_pred['PSA']
    df_pred.dropna(subset=features, inplace=True)

    X = df_pred[features]
    y = df_pred['Result']
    
    scaler_pred = StandardScaler()
    scaler_pred.fit(X)
    
    model_pred = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model_pred.fit(scaler_pred.transform(X), y)

    st.markdown("#### Inserisci i dati della partita")
    
    col1, col2 = st.columns(2)
    with col1:
        home_pos_input = st.number_input("Posizione in classifica squadra di casa:", min_value=1, format="%d")
        odd_home_input = st.number_input("Quota Home:", min_value=1.0, format="%f")
        odd_draw_input = st.number_input("Quota Draw:", min_value=1.0, format="%f")
        
    with col2:
        away_pos_input = st.number_input("Posizione in classifica squadra in trasferta:", min_value=1, format="%d")
        odd_away_input = st.number_input("Quota Away:", min_value=1.0, format="%f")
        st.markdown("<br>", unsafe_allow_html=True) # Spaziatore
        if st.button("Prevedi Risultato"):
            if home_pos_input and odd_home_input and odd_draw_input and away_pos_input and odd_away_input:
                with st.spinner('Calcolo della previsione...'):
                    predictions = predict_single_match(model_pred, scaler_pred, home_pos_input, away_pos_input, odd_home_input, odd_draw_input, odd_away_input)
                    if predictions is not None:
                        st.subheader("Risultato della Previsione")
                        st.metric(label="Probabilità di Vittoria Casa", value=f"{predictions[0]*100:.2f}%")
                        st.metric(label="Probabilità di Pareggio", value=f"{predictions[1]*100:.2f}%")
                        st.metric(label="Probabilità di Vittoria Trasferta", value=f"{predictions[2]*100:.2f}%")
                        
                        st.subheader("Potenziali Value Bet")
                        if (predictions[0] * odd_home_input) > 1:
                            st.success(f"✅ **Value Bet BACK su HOME** (Valore: {predictions[0] * odd_home_input:.2f})")
                        if (predictions[1] * odd_draw_input) > 1:
                            st.success(f"✅ **Value Bet BACK su DRAW** (Valore: {predictions[1] * odd_draw_input:.2f})")
                        if (predictions[2] * odd_away_input) > 1:
                            st.success(f"✅ **Value Bet BACK su AWAY** (Valore: {predictions[2] * odd_away_input:.2f})")
                        
                        if ((predictions[1] + predictions[2]) * (odd_home_input/(odd_home_input-1))) > 1:
                            st.success(f"✅ **Value Bet LAY su HOME** (Valore: {(predictions[1] + predictions[2]) * (odd_home_input/(odd_home_input-1)):.2f})")
                        if ((predictions[0] + predictions[2]) * (odd_draw_input/(odd_draw_input-1))) > 1:
                            st.success(f"✅ **Value Bet LAY su DRAW** (Valore: {(predictions[0] + predictions[2]) * (odd_draw_input/(odd_draw_input-1)):.2f})")
                        if ((predictions[0] + predictions[1]) * (odd_away_input/(odd_away_input-1))) > 1:
                            st.success(f"✅ **Value Bet LAY su AWAY** (Valore: {(predictions[0] + predictions[1]) * (odd_away_input/(odd_away_input-1)):.2f})")

                    else:
                        st.error("Impossibile calcolare la previsione. Controlla i valori inseriti.")

