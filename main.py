import streamlit as st
import psycopg2
import pandas as pd

st.title("📊 Analisi ROI / Winrate / Points (1%) - Back & Lay")

# Funzione di connessione
def run_query(query):
    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        dbname=st.secrets["postgres"]["dbname"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        sslmode="require"
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Menu campionati
leagues = run_query('SELECT DISTINCT league FROM "Matches" ORDER BY league;')
league_list = leagues['league'].dropna().tolist()
selected_league = st.selectbox("Seleziona Campionato", ["Tutti"] + league_list)

# Menu label_odds
label_options = [
    "home strong fav", "home med fav", "home small fav",
    "away strong fav", "away med fav", "away small fav",
    "supercompetitive"
]
selected_label = st.selectbox("Seleziona Label Odds", label_options)

if st.button("Calcola ROI"):
    filtro = ""
    if selected_league != "Tutti":
        filtro += f" AND league = '{selected_league}'"
    if selected_label:
        filtro += f" AND label_odds = '{selected_label}'"

    query = f'''
        SELECT esito, odd_home, odd_draw, odd_away
        FROM "Matches"
        WHERE 1=1 {filtro};
    '''
    df = run_query(query)

    if df.empty:
        st.warning("Nessun dato trovato.")
    else:
        n_bets = len(df)

        def calc_metrics(esito_val, odd_col):
            wins = df[df['esito'] == esito_val]
            winrate = len(wins) / n_bets
            avg_odd = pd.to_numeric(df[odd_col], errors='coerce').mean()

            profit_back = (len(wins) * (avg_odd - 1)) - (n_bets - len(wins))
            roi_back = (profit_back / n_bets) * 100

            profit_lay = ((n_bets - len(wins)) * 1) - (len(wins) * (avg_odd - 1))
            roi_lay = (profit_lay / n_bets) * 100

            return {
                "n_bets": n_bets,
                "winrate": round(winrate * 100, 2),
                "roi_back": round(roi_back, 2),
                "roi_lay": round(roi_lay, 2),
                "points_back": round(profit_back, 2),
                "points_lay": round(profit_lay, 2)
            }

        metrics_1 = calc_metrics("1", "odd_home")
        metrics_X = calc_metrics("X", "odd_draw")
        metrics_2 = calc_metrics("2", "odd_away")

        result_df = pd.DataFrame([
            {"Selezione": "1", **metrics_1},
            {"Selezione": "X", **metrics_X},
            {"Selezione": "2", **metrics_2},
        ])

        st.subheader("Risultati ROI / Winrate / Points (1%)")
        st.dataframe(result_df)
