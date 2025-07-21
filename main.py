import streamlit as st 
import psycopg2
import pandas as pd
import numpy as np

st.title("Matches + Over goals da Risultati Esatti")

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

df = run_query('SELECT * FROM "Matches";')

# Aggiungi colonna risultato_ft
if "gol_home_ft" in df and "gol_away_ft" in df:
    df["risultato_ft"] = df["gol_home_ft"].astype(int).astype(str) + "-" + df["gol_away_ft"].astype(int).astype(str)

# Applica filtri (come già implementato)
# … (mantieni il tuo blocco completo di filtri) …

# Dopo il filtro:
filtered_df = df.copy()  # già filtrato

# Calcolo Over da risultato_ft
filtered_df[["home_g", "away_g"]] = filtered_df["risultato_ft"].str.split("-", expand=True).astype(int)
filtered_df["tot"] = filtered_df["home_g"] + filtered_df["away_g"]

thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
res = []
tot = len(filtered_df)

for t in thresholds:
    count = (filtered_df["tot"] > t).sum()
    pct = round(100 * count / tot, 2) if tot else 0
    odd = round(100 / pct, 2) if pct else "-"
    res.append([f"Over {t}", count, pct, odd])

over_df = pd.DataFrame(res, columns=["Mercato","Conteggio","Percentuale %","Odd Minima"])
st.subheader("Over Goals da Griglia Risultati Esatti")
st.table(over_df)
