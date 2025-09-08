import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Funzione principale per l'analisi del modello (parte 1)
def calculate_probabilities_and_value_bets(df, ranking_option):
    
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
        'Home_Pos_Tot': 'HomePos_Tot',
        'Away_Pos_Tot': 'AwayPos_Tot',
        'Home_Pos_H': 'HomePos_H',
        'Away_Pos_A': 'AwayPos_A'
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
    
    # Selezione delle colonne per la classifica in base all'opzione scelta
    if ranking_option == 'Classifica Totale':
        df['HomePos'] = df['HomePos_Tot']
        df['AwayPos'] = df['AwayPos_Tot']
    else: # Classifica per Casa/Trasferta
        df['HomePos'] = df['HomePos_H']
        df['AwayPos'] = df['AwayPos_A']
    
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
        
        results_df = pd.concat([results_df, pd.DataFrame([{
            'Campionato': league, 'ROI Home Back': roi_home_back, 'ROI Draw Back': roi_draw_back, 'ROI Away Back': roi_away_back,
            'ROI Home Lay': roi_home_lay, 'ROI Draw Lay': roi_draw_lay, 'ROI Away Lay': roi_away_lay
        }])], ignore_index=True)
    
    st.markdown("---")
    st.markdown("## Riepilogo dei Risultati per Campionato")
    st.dataframe(results_df)

# Funzione per prevedere una singola partita (parte 2)
def predict_single_match(model, scaler, home_pos, away_pos, odd_home, odd_draw, odd_away):
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
    ranking_option = st.selectbox(
        "Scegli il tipo di classifica da utilizzare:",
        ('Classifica Totale', 'Classifica per Casa/Trasferta')
    )
    
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip', engine='python')
        st.write("Anteprima dei dati caricati:")
        st.dataframe(df.head())
        
        required_cols = ['League', 'Home_Team', 'Away_Team', 'Gol_Home_FT', 'Gol_Away_FT', 'Odd_Home', 'Odd_Draw', 'Odd__Away', 'Home_Pos_Tot', 'Away_Pos_Tot', 'Home_Pos_H', 'Away_Pos_A']
        if all(col in df.columns for col in required_cols):
            if st.button("Esegui Analisi Storica"):
                with st.spinner('Elaborazione in corso...'):
                    st.session_state.df_historical = df.copy()
                    st.session_state.ranking_option = ranking_option
                    calculate_probabilities_and_value_bets(st.session_state.df_historical, st.session_state.ranking_option)
                st.success('Analisi completata!')
        else:
            missing_cols = [col for col in required_cols if col not in df.columns]
            st.error(f"Il file CSV deve contenere le seguenti colonne. Mancano: {', '.join(missing_cols)}")
    except Exception as e:
        st.error(f"Errore durante la lettura del file: {e}")

st.markdown("---")

st.header("2. Previsione Nuova Partita")

if 'df_historical' in st.session_state and st.session_state.df_historical is not None:
    
    # Preparazione del modello per la previsione
    df_pred = st.session_state.df_historical.rename(columns={
        'League': 'league', 'Home_Team': 'HomeTeam', 'Away_Team': 'AwayTeam',
        'Gol_Home_FT': 'FTHG', 'Gol_Away_FT': 'FTAG', 'Odd_Home': 'PSH',
        'Odd_Draw': 'PSD', 'Odd__Away': 'PSA',
        'Home_Pos_Tot': 'HomePos_Tot', 'Away_Pos_Tot': 'AwayPos_Tot',
        'Home_Pos_H': 'HomePos_H', 'Away_Pos_A': 'AwayPos_A'
    })
    for col in ['PSH', 'PSD', 'PSA']:
        df_pred[col] = df_pred[col].astype(str).str.replace(',', '.', regex=False)
        df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')
        df_pred[col] = df_pred[col].replace(0, np.nan)
    df_pred['Result'] = df_pred.apply(lambda row: 0 if row['FTHG'] > row['FTAG'] else (1 if row['FTHG'] == row['FTAG'] else 2), axis=1)

    if st.session_state.ranking_option == 'Classifica Totale':
        df_pred['HomePos'] = df_pred['HomePos_Tot']
        df_pred['AwayPos'] = df_pred['AwayPos_Tot']
    else:
        df_pred['HomePos'] = df_pred['HomePos_H']
        df_pred['AwayPos'] = df_pred['AwayPos_A']
    
    features = ['HomeOddsProb', 'DrawOddsProb', 'AwayOddsProb', 'HomePos', 'AwayPos']
    df_pred['HomeOddsProb'] = 1 / df_pred['PSH']
    df_pred['DrawOddsProb'] = 1 / df_pred['PSD']
    df_pred['AwayOddsProb'] = 1 / df_pred['PSA']
    df_pred.dropna(subset=features, inplace=True)
    
    # Aggiunge il selettore per il campionato
    available_leagues = df_pred['league'].unique()
    selected_league = st.selectbox("Scegli un campionato per la previsione:", available_leagues)

    # Filtra i dati per il campionato selezionato prima di addestrare il modello
    df_pred_filtered = df_pred[df_pred['league'] == selected_league].copy()
    
    if df_pred_filtered.empty or len(df_pred_filtered) < 10:
        st.warning(f"Dati insufficienti o non validi per il campionato {selected_league}. Scegli un altro campionato.")
    else:
        X = df_pred_filtered[features]
        y = df_pred_filtered['Result']
        
        scaler_pred = StandardScaler()
        scaler_pred.fit(X)
        
        model_pred = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        model_pred.fit(scaler_pred.transform(X), y)
        
        # Calcolo l'accuratezza del modello sul test set per il campionato selezionato
        y_test_pred = model_pred.predict(scaler_pred.transform(X))
        accuracy = accuracy_score(y, y_test_pred)

        st.markdown(f"#### Inserisci i dati per la previsione ({selected_league})")

        col1, col2 = st.columns(2)
        with col1:
            home_pos_input = st.number_input("Posizione in classifica squadra di casa:", min_value=1, format="%d", value=1)
            odd_home_input = st.number_input("Quota Home:", min_value=1.0, format="%f", value=1.0)
            odd_draw_input = st.number_input("Quota Draw:", min_value=1.0, format="%f", value=1.0)
            
        with col2:
            away_pos_input = st.number_input("Posizione in classifica squadra in trasferta:", min_value=1, format="%d", value=1)
            odd_away_input = st.number_input("Quota Away:", min_value=1.0, format="%f", value=1.0)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Prevedi Risultato"):
                if home_pos_input and odd_home_input and odd_draw_input and away_pos_input and odd_away_input:
                    with st.spinner('Calcolo della previsione...'):
                        predictions = predict_single_match(model_pred, scaler_pred, home_pos_input, away_pos_input, odd_home_input, odd_draw_input, odd_away_input)
                        if predictions is not None:
                            st.subheader("Risultato della Previsione")
                            st.metric(label="Probabilità di Vittoria Casa", value=f"{predictions[0]*100:.2f}%")
                            st.metric(label="Probabilità di Pareggio", value=f"{predictions[1]*100:.2f}%")
                            st.metric(label="Probabilità di Vittoria Trasferta", value=f"{predictions[2]*100:.2f}%")
                            
                            st.subheader("Informazioni sul Modello")
                            st.metric(label="Accuratezza del Modello", value=f"{accuracy*100:.2f}%")
                            st.metric(label="Numero di Campioni per l'Addestramento", value=f"{len(df_pred_filtered)}")
                            
                            st.subheader("Quote Reali Calcolate dal Modello")
                            st.metric(label="Quota Reale Home", value=f"{1/predictions[0]:.2f}")
                            st.metric(label="Quota Reale Draw", value=f"{1/predictions[1]:.2f}")
                            st.metric(label="Quota Reale Away", value=f"{1/predictions[2]:.2f}")
                            
                            st.subheader("Analisi della Value Bet per Esito")
                            
                            # Analisi Home
                            st.markdown("**Home Win**")
                            value_back_home = predictions[0] * odd_home_input
                            value_lay_home = 0
                            if odd_home_input > 1:
                                value_lay_home = (predictions[1] + predictions[2]) * (odd_home_input/(odd_home_input-1))
                            
                            if value_back_home > 1 and value_lay_home > 1:
                                if value_back_home > value_lay_home:
                                    st.write(f"✅ **Value Bet in BACK** (Valore: {value_back_home:.2f})")
                                else:
                                    st.write(f"✅ **Value Bet in LAY** (Valore: {value_lay_home:.2f})")
                            elif value_back_home > 1:
                                st.write(f"✅ **Value Bet in BACK** (Valore: {value_back_home:.2f})")
                            elif value_lay_home > 1:
                                st.write(f"✅ **Value Bet in LAY** (Valore: {value_lay_home:.2f})")
                            else:
                                st.info("ℹ️ Nessuna Value Bet trovata per questo esito.")

                            # Analisi Draw
                            st.markdown("**Pareggio**")
                            value_back_draw = predictions[1] * odd_draw_input
                            value_lay_draw = 0
                            if odd_draw_input > 1:
                                value_lay_draw = (predictions[0] + predictions[2]) * (odd_draw_input/(odd_draw_input-1))
                            
                            if value_back_draw > 1 and value_lay_draw > 1:
                                if value_back_draw > value_lay_draw:
                                    st.write(f"✅ **Value Bet in BACK** (Valore: {value_back_draw:.2f})")
                                else:
                                    st.write(f"✅ **Value Bet in LAY** (Valore: {value_lay_draw:.2f})")
                            elif value_back_draw > 1:
                                st.write(f"✅ **Value Bet in BACK** (Valore: {value_back_draw:.2f})")
                            elif value_lay_draw > 1:
                                st.write(f"✅ **Value Bet in LAY** (Valore: {value_lay_draw:.2f})")
                            else:
                                st.info("ℹ️ Nessuna Value Bet trovata per questo esito.")

                            # Analisi Away
                            st.markdown("**Away Win**")
                            value_back_away = predictions[2] * odd_away_input
                            value_lay_away = 0
                            if odd_away_input > 1:
                                value_lay_away = (predictions[0] + predictions[1]) * (odd_away_input/(odd_away_input-1))

                            if value_back_away > 1 and value_lay_away > 1:
                                if value_back_away > value_lay_away:
                                    st.write(f"✅ **Value Bet in BACK** (Valore: {value_back_away:.2f})")
                                else:
                                    st.write(f"✅ **Value Bet in LAY** (Valore: {value_lay_away:.2f})")
                            elif value_back_away > 1:
                                st.write(f"✅ **Value Bet in BACK** (Valore: {value_back_away:.2f})")
                            elif value_lay_away > 1:
                                st.write(f"✅ **Value Bet in LAY** (Valore: {value_lay_away:.2f})")
                            else:
                                st.info("ℹ️ Nessuna Value Bet trovata per questo esito.")
                        else:
                            st.error("Impossibile calcolare la previsione. Controlla i valori inseriti.")
