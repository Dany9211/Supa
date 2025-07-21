import streamlit as st
import psycopg2
import pandas as pd

st.title("Analisi ROI Back e Lay - Football")

home_filter = st.text_input("Filtra per Squadra Home (opzionale)")
away_filter = st.text_input("Filtra per Squadra Away (opzionale)")

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

if st.button("Calcola ROI"):
    try:
        query = 'SELECT * FROM "Matches"'
        filters = []
        if home_filter:
            filters.append(f"squadra_home = '{home_filter}'")
        if away_filter:
            filters.append(f"squadra_away = '{away_filter}'")
        if filters:
            query += " WHERE " + " AND ".join(filters)
        df = run_query(query)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Errore durante l'esecuzione della query: {e}")
