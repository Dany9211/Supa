import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Configurazione database Supabase
DB_CONFIG = {
    "host": "db.xflumlbbsjoyrfgoeeje.supabase.co",
    "port": "5432",
    "database": "postgres",
    "user": "postgres",
    "password": "fickyw-0muxkI-muzvik"
}

def run_query(query):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Errore nella query: {e}")
        return pd.DataFrame()

st.title("📊 Analisi ROI e Winrate - 1X2")

# Input utente
campionato = st.text_input("Filtra per campionato (es. ITA 1)")
squadra_home = st.text_input("Filtra per squadra Home (opzionale)")
squadra_away = st.text_input("Filtra per squadra Away (opzionale)")

if st.button("Calcola ROI"):
    filtro = "WHERE 1=1"
    if campionato:
        filtro += f" AND league = '{campionato}'"
    if squadra_home:
        filtro += f" AND home = '{squadra_home}'"
    if squadra_away:
        filtro += f" AND away = '{squadra_away}'"

    query = f"""
        SELECT label_odds, esito, COUNT(*) as n_bets,
               ROUND(100.0 * SUM(CASE WHEN esito = '1' THEN 1 ELSE 0 END)/COUNT(*),2) as winrate_1,
               ROUND(100.0 * SUM(CASE WHEN esito = 'X' THEN 1 ELSE 0 END)/COUNT(*),2) as winrate_X,
               ROUND(100.0 * SUM(CASE WHEN esito = '2' THEN 1 ELSE 0 END)/COUNT(*),2) as winrate_2
        FROM formatt
        {filtro}
        GROUP BY label_odds, esito
        ORDER BY label_odds;
    """
    df = run_query(query)
    if not df.empty:
        st.dataframe(df)
    else:
        st.warning("Nessun risultato trovato.")
