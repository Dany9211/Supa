
import streamlit as st
import psycopg2
import pandas as pd

def run_query(query):
    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        dbname=st.secrets["postgres"]["dbname"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"]
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

st.title("Analisi ROI Back e Lay - Football")

home = st.text_input("Filtra per Squadra Home (opzionale)")
away = st.text_input("Filtra per Squadra Away (opzionale)")

if st.button("Calcola ROI"):
    try:
        query = "SELECT * FROM partite LIMIT 10"
        df = run_query(query)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Errore durante l'esecuzione della query: {e}")
