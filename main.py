
import streamlit as st
import psycopg2
import pandas as pd

# Connessione al database
def get_connection():
    return psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        dbname=st.secrets["postgres"]["dbname"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"]
    )

def run_query(query):
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)

st.title("Analisi ROI Back e Lay - Football")

# Input opzionali
team_home = st.text_input("Filtra per Squadra Home (opzionale)")
team_away = st.text_input("Filtra per Squadra Away (opzionale)")

if st.button("Calcola ROI"):
    query = "SELECT * FROM formatt LIMIT 10;"  # Esempio: mostra prime 10 righe
    try:
        df = run_query(query)
        st.write(df)
    except Exception as e:
        st.error(f"Errore durante l'esecuzione della query: {e}")
