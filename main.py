import streamlit as st
import psycopg2
import pandas as pd

st.title("Test Connessione e Query - Matches")

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

if st.button("Esegui Test Query"):
    try:
        query = 'SELECT * FROM "Matches" LIMIT 10;'
        df = run_query(query)
        st.success("✅ Connessione e query riuscite!")
        st.dataframe(df)
    except Exception as e:
        st.error(f"❌ Errore durante l'esecuzione della query: {e}")
