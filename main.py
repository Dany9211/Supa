import streamlit as st 
import psycopg2
import pandas as pd
import numpy as np

st.title("Filtro Completo Matches (filtri ottimizzati)")

# Connessione al database
@st.cache_data(allow_output_mutation=True)
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
st.write(f"**Righe totali nel dataset:** {len(df)}")

# Aggiungi colonna risultato_ft
def format_risultato(rh, ra):
    combo = f"{int(rh)}-{int(ra)}"
    allowed = ["0-0","0-1","0-2","0-3","1-0","1-1","1-2","1-3",
               "2-0","2-1","2-2","2-3","3-0","3-1","3-2","3-3"]
    if combo in allowed:
        return combo
    if rh > ra:
        return "Altro risultato casa vince"
    elif rh < ra:
        return "Altro risultato ospite vince"
    else:
        return "Altro pareggio"

df["risultato_ft"] = df.apply(lambda r: format_risultato(r["gol_home_ft"], r["gol_away_ft"]), axis=1)

# Impostazione filtri
filters = {}
dropdown_cols = ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht"]

for col in df.columns:
    if col.lower() in ["id","data"] or "minutaggio" in col.lower() or any(x in col.lower() for x in ["primo","secondo","terzo","quarto","quinto"]):
        continue
    if col in dropdown_cols:
        vals = sorted(df[col].dropna().unique().astype(int))
        if 0 not in vals:
            vals = [0] + vals
        sel = st.selectbox(f"Filtra per {col}", ["Tutti"] + [str(v) for v in vals])
        if sel != "Tutti":
            filters[col] = int(sel)
    else:
        tmp = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
        if tmp.notnull().sum() > 0:
            vmin, vmax = tmp.min(), tmp.max()
            sel_range = st.slider(f"Filtro per {col}", float(vmin), float(vmax), (float(vmin), float(vmax)), step=0.01)
            filters[col] = sel_range
        else:
            opts = df[col].dropna().unique().tolist()
            sel = st.selectbox(f"{col} (opzionale)", ["Tutti"] + [str(x) for x in opts])
            if sel != "Tutti":
                filters[col] = sel

filtered = df.copy()
for col, val in filters.items():
    if isinstance(val, tuple):
        low, high = val
        tmp = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
        filtered = filtered[(tmp >= low) & (tmp <= high)]
    else:
        filtered = filtered[filtered[col].astype(str) == str(val)]

st.subheader("Dati Filtrati")
st.dataframe(filtered)
st.write(f"**Righe visualizzate:** {len(filtered)}")

# Distribuzione risultati esatti
allowed = ["0-0","0-1","0-2","0-3","1-0","1-1","1-2","1-3","2-0","2-1","2-2","2-3","3-0","3-1","3-2","3-3",
           "Altro risultato casa vince","Altro pareggio","Altro risultato ospite vince"]
dist = filtered["risultato_ft"].value_counts().reindex(allowed, fill_value=0).reset_index()
dist.columns = ["Risultato", "Conteggio"]
dist["Percentuale %"] = (dist["Conteggio"] / len(filtered) * 100).round(2)
st.subheader("Distribuzione Risultati Esatti")
st.dataframe(dist)

# Calcolo WinRate e ROI 1×2
filtered["esito"] = filtered.apply(lambda r: "1" if r["gol_home_ft"]>r["gol_away_ft"] 
                                   else ("2" if r["gol_home_ft"]<r["gol_away_ft"] else "X"), axis=1)
n = len(filtered)
esiti = filtered["esito"].value_counts().reindex(["1","X","2"], fill_value=0)
winrate = (esiti / n *100).round(2)

# Profit (stake 1%) e ROI%
profit = {
    e: (esiti[e] * (filtered.loc[filtered["esito"]==e, f"odd_{e}"].mean() - 1) - (n - esiti[e])) * 0.01
    for e in ["1","X","2"]
}
roi = {e: round(profit[e] / (n*0.01) * 100, 2) for e in profit}

summary = pd.DataFrame({
    "N. Bets": esiti,
    "WinRate %": winrate,
    "Profit (1%)": [round(profit[e],2) for e in ["1","X","2"]],
    "ROI %": [roi[e] for e in ["1","X","2"]]
}, index=["Home ⬜", "Pareggio ⚪", "Trasferta ☐"])
st.subheader("WinRate & ROI (1% stake)")
st.dataframe(summary)
