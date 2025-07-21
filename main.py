import streamlit as st
import psycopg2
import pandas as pd

# --- CONFIGURAZIONE DB SUPABASE ---
DB_CONFIG = {
    "host": "db.xflumlbbsjoyrfgoeeje.supabase.co",
    "port": "5432",
    "database": "postgres",
    "user": "postgres",
    "password": "fickyw-0muxkI-muzvik"
}

def run_query(query):
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

st.title("Analisi ROI Back e Lay - Football")

# Filtri
league = st.text_input("Inserisci Campionato (es. ITA 1)", "")
label = st.text_input("Inserisci Label Odds (es. home strong fav)", "")
home_team = st.text_input("Filtra per Squadra Home (opzionale)", "")
away_team = st.text_input("Filtra per Squadra Away (opzionale)", "")

if st.button("Calcola ROI"):
    filters = []
    if league:
        filters.append(f"league = '{league}'")
    if label:
        filters.append(f"label_odds = '{label}'")
    if home_team:
        filters.append(f"home = '{home_team}'")
    if away_team:
        filters.append(f"away = '{away_team}'")
    where_clause = "WHERE " + " AND ".join(filters) if filters else ""

    query = f"""
    WITH matches AS (
        SELECT *,
            CASE
                WHEN CAST(REPLACE(odd_home, ',', '.') AS numeric) BETWEEN 1.01 AND 1.50 THEN 'home strong fav'
                WHEN CAST(REPLACE(odd_home, ',', '.') AS numeric) BETWEEN 1.51 AND 2.00 THEN 'home med fav'
                WHEN CAST(REPLACE(odd_home, ',', '.') AS numeric) BETWEEN 2.01 AND 2.50 THEN 'home small fav'
                WHEN CAST(REPLACE(odd_away, ',', '.') AS numeric) BETWEEN 1.01 AND 1.50 THEN 'away strong fav'
                WHEN CAST(REPLACE(odd_away, ',', '.') AS numeric) BETWEEN 1.51 AND 2.00 THEN 'away med fav'
                WHEN CAST(REPLACE(odd_away, ',', '.') AS numeric) BETWEEN 2.01 AND 2.50 THEN 'away small fav'
                WHEN CAST(REPLACE(odd_home, ',', '.') AS numeric) < 3.00 
                     AND CAST(REPLACE(odd_away, ',', '.') AS numeric) < 3.00 THEN 'supercompetitive'
            END AS label_odds
        FROM formatt
        {where_clause}
    )
    SELECT
        label_odds,
        COUNT(*) AS n_bets,
        SUM(CASE WHEN esito = '1' THEN CAST(REPLACE(odd_home, ',', '.') AS numeric) - 1 ELSE -1 END) AS back_1,
        SUM(CASE WHEN esito = 'X' THEN CAST(REPLACE(odd_draw, ',', '.') AS numeric) - 1 ELSE -1 END) AS back_x,
        SUM(CASE WHEN esito = '2' THEN CAST(REPLACE(odd_away, ',', '.') AS numeric) - 1 ELSE -1 END) AS back_2,
        SUM(CASE WHEN esito = '1' THEN -(CAST(REPLACE(odd_home, ',', '.') AS numeric) * 1.03 - 1) ELSE 1 END) AS lay_1,
        SUM(CASE WHEN esito = 'X' THEN -(CAST(REPLACE(odd_draw, ',', '.') AS numeric) * 1.03 - 1) ELSE 1 END) AS lay_x,
        SUM(CASE WHEN esito = '2' THEN -(CAST(REPLACE(odd_away, ',', '.') AS numeric) * 1.03 - 1) ELSE 1 END) AS lay_2
    FROM matches
    GROUP BY label_odds
    ORDER BY label_odds;
    """

    df = run_query(query)
    if not df.empty:
        for col in ['back_1', 'back_x', 'back_2', 'lay_1', 'lay_x', 'lay_2']:
            df[f'{col}_roi'] = round((df[col] / df['n_bets']) * 100, 2)
            df[f'{col}_pts'] = round(df[col], 2)
        st.write("### Risultati ROI e Profit Points")
        st.dataframe(df)
    else:
        st.warning("Nessun dato trovato con i filtri selezionati.")
