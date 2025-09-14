
import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Analisi Pattern Gol", layout="wide")
st.title("Analisi Pattern Gol")

# =====================================================================================
# Helpers
# =====================================================================================

TIMEBANDS = {
    "0-10": (0, 10),
    "11-20": (11, 20),
    "21-30": (21, 30),
    "31-39": (31, 39),
    "40-45": (40, 45),
    "45+": (45, 45),   # per calcoli HT considerato come 45
    "46-60": (46, 60),
    "61-75": (61, 75),
    "76-90": (76, 90),
    "90+": (90, 90),   # per calcoli FT considerato come 90
}

GOAL_COLS = dict(
    h_tim="home_team_goal_timings",
    a_tim="away_team_goal_timings",
    ht_h="home_team_goal_count_half_time",
    ht_a="away_team_goal_count_half_time",
    ft_h="home_team_goal_count",
    ft_a="away_team_goal_count",
)

META_COLS = ["date_GMT","league","home_team_name","away_team_name"]

def parse_timings(timing_str: str):
    """
    Converte stringhe come '6,45+1,67' in lista di minuti interi.
    Logica:
      - 'M+X' -> usa solo M per i calcoli (45+1 => 45; 90+2 => 90)
      - 'M'   -> int(M)
    """
    if pd.isna(timing_str) or str(timing_str).strip() == "":
        return []
    out = []
    for token in str(timing_str).split(","):
        token = token.strip()
        if not token:
            continue
        # accetta formati tipo 45+1 o 45'1
        base = token
        if "+" in token:
            base = token.split("+", 1)[0]
        elif "'" in token:
            base = token.split("'", 1)[0]
        try:
            out.append(int(base))
        except:
            # ignora token strani
            pass
    return sorted(out)

def cum_score_at_min(row, minute: int):
    """
    Restituisce (gol_home_fino_a_min, gol_away_fino_a_min), inclusivo.
    Nota: 45+ conta come 45; 90+ come 90.
    """
    h_list = parse_timings(row.get(GOAL_COLS["h_tim"], ""))
    a_list = parse_timings(row.get(GOAL_COLS["a_tim"], ""))
    h = sum(1 for m in h_list if m <= minute)
    a = sum(1 for m in a_list if m <= minute)
    return h, a

def goals_between(row, start_min: int, end_min: int):
    """
    Conta i gol (H,A) nel range inclusivo [start_min, end_min].
    """
    h_list = parse_timings(row.get(GOAL_COLS["h_tim"], ""))
    a_list = parse_timings(row.get(GOAL_COLS["a_tim"], ""))
    h = sum(1 for m in h_list if start_min <= m <= end_min)
    a = sum(1 for m in a_list if start_min <= m <= end_min)
    return h, a

def check_first_goal_enhanced(row, fh, fa, min_first, max_first):
    """
    Verifica che il primo gol (cumulativo) porti lo score a (fh,fa) e sia nel timeband [min_first, max_first] inclusivo.
    Il "primo gol" qui è inteso come primo gol della partita che raggiunge esattamente quello score.
    """
    h_list = parse_timings(row.get(GOAL_COLS["h_tim"], ""))
    a_list = parse_timings(row.get(GOAL_COLS["a_tim"], ""))
    all_g = [(m, "H") for m in h_list] + [(m, "A") for m in a_list]
    all_g.sort(key=lambda x: x[0])
    h=a=0
    for m, side in all_g:
        if side == "H":
            h += 1
        else:
            a += 1
        if (h, a) == (fh, fa):
            return (min_first <= m <= max_first)
    return False

def check_second_goal_enhanced(row, sh, sa, min_second, max_second):
    """
    Verifica che il secondo gol (che porta a score sh-sa) sia nel timeband inclusivo.
    """
    h_list = parse_timings(row.get(GOAL_COLS["h_tim"], ""))
    a_list = parse_timings(row.get(GOAL_COLS["a_tim"], ""))
    all_g = [(m, "H") for m in h_list] + [(m, "A") for m in a_list]
    all_g.sort(key=lambda x: x[0])
    h=a=0
    for m, side in all_g:
        if side == "H":
            h += 1
        else:
            a += 1
        if (h, a) == (sh, sa):
            return (min_second <= m <= max_second)
    return False

def df_cols_ok(df: pd.DataFrame):
    base = [GOAL_COLS["h_tim"], GOAL_COLS["a_tim"]]
    return all(c in df.columns for c in base)

# =====================================================================================
# Sidebar: upload e filtri base
# =====================================================================================
st.sidebar.header("Dati")
uploaded = st.sidebar.file_uploader("Carica CSV", type=["csv"])
sep = st.sidebar.radio("Separatore", options=[",",";"], index=1, horizontal=True)

if uploaded is not None:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, sep=sep, encoding="utf-8", engine="c", low_memory=False)
else:
    df = pd.DataFrame(columns=META_COLS + list(GOAL_COLS.values()))

if not df.empty:
    leagues = sorted(df["league"].dropna().unique()) if "league" in df.columns else []
    default_leagues = leagues[:1] if leagues else []
    sel_leagues = st.sidebar.multiselect("League", leagues, default=default_leagues)
    if sel_leagues and "league" in df.columns:
        df = df[df["league"].isin(sel_leagues)].copy()

st.caption(f"Righe correnti (post filtri sidebar): {len(df)}")

if not df_cols_ok(df):
    st.warning("Colonne dei timing gol non trovate. Servono 'home_team_goal_timings' e 'away_team_goal_timings'.")

# =====================================================================================
# Pattern input
# =====================================================================================
st.markdown("---")
st.subheader("Vincoli dell'analisi (Pattern)")

colA, colB, colC = st.columns(3)
with colA:
    start_min_patt = st.slider("Minuto vincolo 1", 1, 90, 46)
with colB:
    risultato_attuale_patt = st.text_input("Risultato al minuto (vincolo 1, X-Y)", value="0-0")
with colC:
    use_second_constraint = st.checkbox("Aggiungi vincolo 2 (minuto + risultato)", value=False)

if use_second_constraint:
    colD, colE = st.columns(2)
    with colD:
        second_min_patt = st.slider("Minuto vincolo 2", 1, 90, max(start_min_patt, 46))
    with colE:
        risultato_attuale_patt2 = st.text_input("Risultato al minuto (vincolo 2, X-Y)", value=risultato_attuale_patt)
else:
    second_min_patt = None
    risultato_attuale_patt2 = None

st.markdown("**Filtri opzionali su 1°/2° gol:**")
colF, colG = st.columns(2)
with colF:
    first_goal_result = st.selectbox("Risultato dopo 1° gol", ["Nessun Filtro","1-0","0-1"], index=0)
    first_goal_time = st.selectbox("Timeband del 1° gol", ["Nessun Filtro"] + list(TIMEBANDS.keys()), index=0)
with colG:
    second_goal_result = st.selectbox("Risultato dopo 2° gol", ["Nessun Filtro","2-0","0-2","1-1"], index=0)
    second_goal_time = st.selectbox("Timeband del 2° gol", ["Nessun Filtro"] + list(TIMEBANDS.keys()), index=0)

st.markdown("---")

# =====================================================================================
# Avvio analisi (button)
# =====================================================================================
if st.button("Avvia Analisi Pattern Gol"):
    if df.empty or not df_cols_ok(df):
        st.error("Dataset vuoto o colonne timing mancanti.")
        st.stop()

    # Gate 1: risultato attuale (fa fede) al minuto selezionato
    try:
        curr_h, curr_a = map(int, str(risultato_attuale_patt).split("-"))
    except Exception:
        st.error("Formato risultato (vincolo 1) non valido. Usa 'X-Y' (es. '0-2').")
        st.stop()

    def match_score_at_min(row):
        h_at_m, a_at_m = cum_score_at_min(row, start_min_patt)
        return (h_at_m == curr_h) and (a_at_m == curr_a)

    df_gate1 = df[df.apply(match_score_at_min, axis=1)]

    # Coerenza HT: se parti dal 46+ => HT deve essere uguale al risultato richiesto al vincolo 1
    try:
        if start_min_patt >= 46 and {GOAL_COLS["ht_h"], GOAL_COLS["ht_a"]}.issubset(df_gate1.columns):
            df_gate1 = df_gate1[
                (df_gate1[GOAL_COLS["ht_h"]].astype("Int64").fillna(0).astype(int) == curr_h) &
                (df_gate1[GOAL_COLS["ht_a"]].astype("Int64").fillna(0).astype(int) == curr_a)
            ]
    except Exception:
        pass

    # Coerenza HT nel primo tempo (richiede uguaglianza, non >=)
    try:
        if start_min_patt <= 45 and {GOAL_COLS["ht_h"], GOAL_COLS["ht_a"]}.issubset(df_gate1.columns):
            df_gate1 = df_gate1[
                (df_gate1[GOAL_COLS["ht_h"]].astype("Int64").fillna(0).astype(int) == curr_h) &
                (df_gate1[GOAL_COLS["ht_a"]].astype("Int64").fillna(0).astype(int) == curr_a)
            ]
    except Exception:
        pass

    # Gate 2 (opzionale): secondo vincolo (minuto + risultato)
    df_gate2 = df_gate1.copy()
    if use_second_constraint and second_min_patt is not None and risultato_attuale_patt2:
        try:
            curr2_h, curr2_a = map(int, str(risultato_attuale_patt2).split("-"))
            def match_score_at_min2(row):
                h2, a2 = cum_score_at_min(row, second_min_patt)
                return (h2 == curr2_h) and (a2 == curr2_a)
            df_gate2 = df_gate2[df_gate2.apply(match_score_at_min2, axis=1)]

            # Se attraversi l'intervallo senza cambio punteggio, imponi: nessun gol fino a 45 e HT uguale
            if start_min_patt <= 45 <= second_min_patt and {GOAL_COLS["ht_h"], GOAL_COLS["ht_a"]}.issubset(df_gate2.columns):
                def no_goals_until_ht_and_ht_equals(row):
                    g_h, g_a = goals_between(row, start_min_patt, 45)
                    ht_ok = (int(row[GOAL_COLS["ht_h"]]) == curr_h) and (int(row[GOAL_COLS["ht_a"]]) == curr_a)
                    return (g_h == 0 and g_a == 0 and ht_ok)
                df_gate2 = df_gate2[df_gate2.apply(no_goals_until_ht_and_ht_equals, axis=1)]
        except Exception:
            pass

    # Filtri 1°/2° gol (opzionali)
    df_after_first = df_gate2.copy()
    try:
        if first_goal_result != "Nessun Filtro" and first_goal_time != "Nessun Filtro":
            fh, fa = map(int, first_goal_result.split("-"))
            min_first, max_first = TIMEBANDS[first_goal_time]
            df_after_first = df_after_first[df_after_first.apply(lambda row: check_first_goal_enhanced(row, fh, fa, min_first, max_first), axis=1)]
    except Exception:
        pass

    df_after_second = df_after_first.copy()
    try:
        if second_goal_result != "Nessun Filtro" and second_goal_time != "Nessun Filtro":
            sh, sa = map(int, second_goal_result.split("-"))
            min_second, max_second = TIMEBANDS[second_goal_time]
            df_after_second = df_after_first[df_after_first.apply(lambda row: check_second_goal_enhanced(row, sh, sa, min_second, max_second), axis=1)]
            # Se il 2° gol è nel 1° tempo, sigilla HT = risultato dopo 2° gol
            if max_second <= 45 and {GOAL_COLS["ht_h"], GOAL_COLS["ht_a"]}.issubset(df_after_second.columns):
                df_after_second = df_after_second[
                    (df_after_second[GOAL_COLS["ht_h"]].astype("Int64").fillna(0).astype(int) == sh) &
                    (df_after_second[GOAL_COLS["ht_a"]].astype("Int64").fillna(0).astype(int) == sa)
                ]
    except Exception:
        pass

    # Risultato finale del pattern
    df_out = df_after_second.copy()

    # Salva in session_state per le statistiche
    st.session_state["df_out"] = df_out.copy()

    st.success(f"Partite trovate: {len(df_out)}")
    cols_to_show = [c for c in META_COLS + [GOAL_COLS["h_tim"],GOAL_COLS["a_tim"],GOAL_COLS["ht_h"],GOAL_COLS["ht_a"],GOAL_COLS["ft_h"],GOAL_COLS["ft_a"]] if c in df_out.columns]
    if cols_to_show:
        st.dataframe(df_out[cols_to_show].reset_index(drop=True).head(200), use_container_width=True)
    else:
        st.dataframe(df_out.head(200), use_container_width=True)

# =====================================================================================
# Statistiche (usano session_state["df_out"])
# =====================================================================================
st.markdown("---")
st.subheader("📊 Statistiche dal sottoinsieme corrente")

df_out = st.session_state.get("df_out", pd.DataFrame())

if df_out is None or df_out.empty:
    st.info("Nessun dato ancora disponibile: premi **Avvia Analisi Pattern Gol** per popolare le statistiche.")
else:
    def _pct(n, d):
        return (100.0 * n / d) if d else 0.0

    def _sign(x):
        return 0 if x == 0 else (1 if x > 0 else -1)

    tot_rows = len(df_out)

    # FT goals (preferisci conteggi diretti se presenti)
    if {GOAL_COLS["ft_h"], GOAL_COLS["ft_a"]}.issubset(df_out.columns):
        ft_h = df_out[GOAL_COLS["ft_h"]].astype("Int64").fillna(0).astype(int)
        ft_a = df_out[GOAL_COLS["ft_a"]].astype("Int64").fillna(0).astype(int)
    else:
        ft_h = df_out[GOAL_COLS["h_tim"]].apply(lambda s: len(parse_timings(s)))
        ft_a = df_out[GOAL_COLS["a_tim"]].apply(lambda s: len(parse_timings(s)))

    # HT goals
    if {GOAL_COLS["ht_h"], GOAL_COLS["ht_a"]}.issubset(df_out.columns):
        ht_h = df_out[GOAL_COLS["ht_h"]].astype("Int64").fillna(0).astype(int)
        ht_a = df_out[GOAL_COLS["ht_a"]].astype("Int64").fillna(0).astype(int)
    else:
        def count_ht(s):
            return sum(1 for m in parse_timings(s) if m <= 45)
        ht_h = df_out[GOAL_COLS["h_tim"]].apply(count_ht)
        ht_a = df_out[GOAL_COLS["a_tim"]].apply(count_ht)

    # SH goals (secondo tempo)
    sh_h = ft_h - ht_h
    sh_a = ft_a - ht_a

    ft_tot = ft_h + ft_a
    ht_tot = ht_h + ht_a
    sh_tot = sh_h + sh_a

    # Toggles
    col1, col2, col3 = st.columns(3)
    with col1:
        show_ft = st.checkbox("Mostra FT", value=True, key="show_ft_stats")
    with col2:
        show_ht = st.checkbox("Mostra HT", value=True, key="show_ht_stats")
    with col3:
        show_sh = st.checkbox("Mostra 2°T (SH)", value=False, key="show_sh_stats")

    # Helpers over
    def over_table(series, thresholds):
        rows = []
        for th in thresholds:
            n = int((series > th).sum())
            rows.append({"Soglia": f"Over {th+0.5}", "N": n, "Win%": round(_pct(n, tot_rows), 2)})
        return pd.DataFrame(rows)

    # FT
    if show_ft:
        with st.expander("FT — Over winrate & 1X2", expanded=False):
            st.write(over_table(ft_tot, [0,1,2,3,4]))
            ft_sign = (ft_h - ft_a).apply(_sign)
            n_home = int((ft_sign > 0).sum())
            n_draw = int((ft_sign == 0).sum())
            n_away = int((ft_sign < 0).sum())
            st.write(pd.DataFrame([
                {"Esito": "1", "N": n_home, "Win%": round(_pct(n_home, tot_rows),2)},
                {"Esito": "X", "N": n_draw, "Win%": round(_pct(n_draw, tot_rows),2)},
                {"Esito": "2", "N": n_away, "Win%": round(_pct(n_away, tot_rows),2)},
            ]))
            # Top risultati esatti FT
            from collections import Counter
            exact_ft = Counter(zip(ft_h, ft_a))
            top_ft = [{"Score": f"{h}-{a}", "N": n, "Win%": round(_pct(n, tot_rows),2)} for (h,a), n in exact_ft.most_common(10)]
            st.markdown("**Top 10 risultati esatti FT**")
            st.write(pd.DataFrame(top_ft))

    # HT
    if show_ht:
        with st.expander("HT — Over winrate & 1X2", expanded=False):
            st.write(over_table(ht_tot, [0,1]))
            ht_sign = (ht_h - ht_a).apply(_sign)
            hth = int((ht_sign > 0).sum())
            htx = int((ht_sign == 0).sum())
            hta = int((ht_sign < 0).sum())
            st.write(pd.DataFrame([
                {"Esito": "1", "N": hth, "Win%": round(_pct(hth, tot_rows),2)},
                {"Esito": "X", "N": htx, "Win%": round(_pct(htx, tot_rows),2)},
                {"Esito": "2", "N": hta, "Win%": round(_pct(hta, tot_rows),2)},
            ]))
            from collections import Counter
            exact_ht = Counter(zip(ht_h, ht_a))
            top_ht = [{"Score": f"{h}-{a}", "N": n, "Win%": round(_pct(n, tot_rows),2)} for (h,a), n in exact_ht.most_common(10)]
            st.markdown("**Top 10 risultati esatti HT**")
            st.write(pd.DataFrame(top_ht))

    # 2° Tempo
    if show_sh:
        with st.expander("2° Tempo — Over winrate & 1X2", expanded=False):
            st.write(over_table(sh_tot, [0,1,2]))
            sh_sign = (sh_h - sh_a).apply(_sign)
            shh = int((sh_sign > 0).sum())
            shx = int((sh_sign == 0).sum())
            sha = int((sh_sign < 0).sum())
            st.write(pd.DataFrame([
                {"Esito": "1", "N": shh, "Win%": round(_pct(shh, tot_rows),2)},
                {"Esito": "X", "N": shx, "Win%": round(_pct(shx, tot_rows),2)},
                {"Esito": "2", "N": sha, "Win%": round(_pct(sha, tot_rows),2)},
            ]))
            from collections import Counter
            exact_sh = Counter(zip(sh_h, sh_a))
            top_sh = [{"Score": f"{h}-{a}", "N": n, "Win%": round(_pct(n, tot_rows),2)} for (h,a), n in exact_sh.most_common(10)]
            st.markdown("**Top 10 risultati esatti 2° Tempo**")
            st.write(pd.DataFrame(top_sh))

# =====================================================================================
# TIMEBAND (inclusivi; 45+ e 90+ separati SOLO nella vista)
# =====================================================================================
st.markdown("---")
st.subheader("📈 Timeband gol (limiti inclusivi; 45+ e 90+ separati in vista)")

df_out = st.session_state.get("df_out", pd.DataFrame())
if not df_out.empty and {"home_team_goal_timings","away_team_goal_timings"}.issubset(df_out.columns):
    _BANDS = ["0-10","11-20","21-30","31-39","40-45","45+","46-60","61-75","76-90","90+"]

    def _band_for_min(m:int) -> str:
        if   0 <= m <= 10:  return "0-10"
        elif 11 <= m <= 20: return "11-20"
        elif 21 <= m <= 30: return "21-30"
        elif 31 <= m <= 39: return "31-39"
        elif 40 <= m <= 45: return "40-45"
        elif 46 <= m <= 60: return "46-60"
        elif 61 <= m <= 75: return "61-75"
        elif 76 <= m <= 90: return "76-90"
        else:               return None

    def _split_tokens(s:str):
        if pd.isna(s) or str(s).strip()=="":
            return []
        toks = []
        for t in str(s).split(","):
            t = t.strip()
            if t:
                toks.append(t)
        return toks

    def _count_row_timebands(home_raw:str, away_raw:str):
        counts_home = {b:0 for b in _BANDS}
        counts_away = {b:0 for b in _BANDS}

        def _accumulate(tokens, is_home:bool):
            for tok in tokens:
                base = tok
                plus = False
                if "+" in tok:
                    base, _ = tok.split("+", 1)
                    plus = True
                elif "'" in tok:
                    base, _ = tok.split("'", 1)
                    plus = True
                try:
                    m = int(base)
                except:
                    continue

                if plus and m == 45:
                    band = "45+"
                elif plus and m == 90:
                    band = "90+"
                else:
                    band = _band_for_min(m)

                if band is None:
                    continue
                if is_home:
                    counts_home[band] += 1
                else:
                    counts_away[band] += 1

        _accumulate(_split_tokens(home_raw), True)
        _accumulate(_split_tokens(away_raw), False)
        return counts_home, counts_away

    tot_home = {b:0 for b in _BANDS}
    tot_away = {b:0 for b in _BANDS}

    for _, r in df_out[["home_team_goal_timings","away_team_goal_timings"]].iterrows():
        ch, ca = _count_row_timebands(r["home_team_goal_timings"], r["away_team_goal_timings"])
        for b in _BANDS:
            tot_home[b] += ch[b]
            tot_away[b] += ca[b]

    df_timeband_home = pd.DataFrame({
        "Timeband": _BANDS,
        "GF (Home)": [tot_home[b] for b in _BANDS],
        "GS (Home)": [tot_away[b] for b in _BANDS],
    })

    df_timeband_away = pd.DataFrame({
        "Timeband": _BANDS,
        "GF (Away)": [tot_away[b] for b in _BANDS],
        "GS (Away)": [tot_home[b] for b in _BANDS],
    })

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(df_timeband_home, use_container_width=True)
    with c2:
        st.dataframe(df_timeband_away, use_container_width=True)
else:
    st.caption("Carica e filtra i dati, poi avvia l'analisi per vedere le timeband.")
