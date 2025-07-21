import streamlit as st
import psycopg2
import pandas as pd

DB_CONFIG = {
    "host": st.secrets["postgres"]["host"],
    "port": st.secrets["postgres"]["port"],
    "dbname": st.secrets["postgres"]["dbname"],
    "user": st.secrets["postgres"]["user"],
    "password": st.secrets["postgres"]["password"],
    "sslmode": "require"
}

def run_query(query):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return pd.DataFrame(data, columns=colnames)

st.title("Analisi ROI Back e Lay - Football")

home_team = st.text_input("Filtra per Squadra Home (opzionale)")
away_team = st.text_input("Filtra per Squadra Away (opzionale)")

if st.button("Calcola ROI"):
    query = "SELECT * FROM matches"
    filters = []
    if home_team:
        filters.append(f"home_team = '{home_team}'")
    if away_team:
        filters.append(f"away_team = '{away_team}'")
    if filters:
        query += " WHERE " + " AND ".join(filters)
    df = run_query(query)
    st.dataframe(df)
