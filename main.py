import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

# =======================
# Parser minuti gol
# =======================
GOAL_TOKEN_RE = re.compile(r'\d+(?:\+\d+)?')

def _parse_goal_tokens(s: str, mapping: str = "base_minute"):
    """
    Estrae minuti da stringhe tipo '12, 45+1, 90+3'.
    mapping:
      - "base_minute": 45+1 -> 45, 90+3 -> 90 (USATO per tutti i calcoli tranne la tabella timeband)
      - "next_minute": 45+1 -> 46, 90+3 -> 93 (NON usato qui; la timeband ha bucket 45+ / 90+ dedicati)
    Ritorna lista ordinata di int.
    """
    tokens = GOAL_TOKEN_RE.findall(str(s))
    minutes = []
    for tok in tokens:
        if '+' in tok:
            base, extra = tok.split('+')
            base = int(base); extra = int(extra)
            if mapping == "next_minute":
                minutes.append(base + max(extra, 1))
            else:
                minutes.append(base)
        else:
            minutes.append(int(tok))
    return sorted(minutes)

def get_goal_minutes(row, side: str, mapping: str = "base_minute"):
    col = "home_team_goal_timings" if side == "home" else "away_team_goal_timings"
    return _parse_goal_tokens(row.get(col, ""), mapping=mapping)

def cum_score_at_min(row, m: int):
    """Ritorna (home, away) cumulati fino al minuto m (inclusivo), con 45+x trattato come 45."""
    gh = get_goal_minutes(row, "home", mapping="base_minute")
    ga = get_goal_minutes(row, "away", mapping="base_minute")
    h = sum(1 for g in gh if g <= m)
    a = sum(1 for g in ga if g <= m)
    return h, a

def goals_between(row, a_exclusive: int, b_inclusive: int):
    """Numero gol in (a_exclusive, b_inclusive] per ciascuna squadra (base_minute)."""
    gh = get_goal_minutes(row, "home", mapping="base_minute")
    ga = get_goal_minutes(row, "away", mapping="base_minute")
    h = sum(1 for g in gh if a_exclusive < g <= b_inclusive)
    a = sum(1 for g in ga if a_exclusive < g <= b_inclusive)
    return h, a

# =======================
# Funzioni pattern PRIMO/SECONDO gol (nessun controllo HT/FT qui)
# =======================
def check_first_goal_enhanced(row, first_home_score, first_away_score, min_first, max_first):
    gol_home = get_goal_minutes(row, "home", mapping="base_minute")
    gol_away = get_goal_minutes(row, "away", mapping="base_minute")
    all_goals = []
    if gol_home: all_goals.extend([(t, 'home') for t in gol_home])
    if gol_away: all_goals.extend([(t, 'away') for t in gol_away])
    all_goals.sort()
    if not all_goals:
        return False
    first_min = all_goals[0][0]
    h_at = sum(1 for g in all_goals if g[1] == 'home' and g[0] <= first_min)
    a_at = sum(1 for g in all_goals if g[1] == 'away' and g[0] <= first_min)
    return (h_at == first_home_score) and (a_at == first_away_score) and (min_first <= first_min <= max_first)

def check_second_goal_enhanced(row, second_home_score, second_away_score, min_second, max_second):
    gol_home = get_goal_minutes(row, "home", mapping="base_minute")
    gol_away = get_goal_minutes(row, "away", mapping="base_minute")
    all_goals = []
    if gol_home: all_goals.extend([(t, 'home') for t in gol_home])
    if gol_away: all_goals.extend([(t, 'away') for t in gol_away])
    all_goals.sort()
    if len(all_goals) < 2:
        return False
    second_min = all_goals[1][0]
    h_at = sum(1 for g in all_goals if g[1] == 'home' and g[0] <= second_min)
    a_at = sum(1 for g in all_goals if g[1] == 'away' and g[0] <= second_min)
    return (h_at == second_home_score) and (a_at == second_away_score) and (min_second <= second_min <= max_second)

# =======================
# Utility formattazione
# =======================
def odd_min_from_percent(p: float):
    if p and p > 0:
        return round(100.0 / p, 2)
    return np.nan

def style_table(df: pd.DataFrame, percent_cols):
    if df is None or len(df) == 0:
        return pd.DataFrame().style
    if isinstance(percent_cols, str):
        percent_cols = [percent_cols]
    fmt = {col: "{:.2f}%" for col in percent_cols}
    fmt.update({
        "Odd Minima": lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        "Odd Minima >= 2 Gol": lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        "Conteggio": "{:,.0f}",
        "Partite con Gol": "{:,.0f}",
    })
    styler = (df.style
                .format(fmt, na_rep="-")
                .background_gradient(subset=[c for c in percent_cols if c in df.columns], cmap="RdYlGn")
                .set_properties(**{"text-align": "center"})
                .set_table_styles([{ 'selector': 'th', 'props': 'text-align: center;' }])
            )
    return styler

# =======================
# Calcoli statistiche
# =======================
def calcola_winrate(df, col_risultato_home, col_risultato_away):
    if df.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
    df_valid = df[(df[col_risultato_home].notna()) & (df[col_risultato_away].notna())].copy()
    if df_valid.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
    risultati = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
    for _, row in df_valid.iterrows():
        try:
            home, away = int(row[col_risultato_home]), int(row[col_risultato_away])
            if home > away: risultati["1 (Casa)"] += 1
            elif home < away: risultati["2 (Trasferta)"] += 1
            else: risultati["X (Pareggio)"] += 1
        except:
            continue
    totale = len(df_valid)
    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale) * 100, 2) if totale > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        stats.append((esito, count, perc, odd_min))
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])

def calcola_over_under(df_to_analyze, period):
    if df_to_analyze.empty:
        return (pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]),
                pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))
    if period == 'ft':
        goals_col = "total_goals_at_full_time"
    elif period == 'ht':
        goals_col = "total_goals_at_half_time"
    elif period == 'sh':
        df_to_analyze = df_to_analyze.copy()
        df_to_analyze['total_goals_at_second_half'] = df_to_analyze['total_goals_at_full_time'] - df_to_analyze['total_goals_at_half_time']
        goals_col = "total_goals_at_second_half"
    else:
        return pd.DataFrame(), pd.DataFrame()
    total_matches = len(df_to_analyze)
    goal_lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    over_data, under_data = [], []
    for gl in goal_lines:
        oc = (df_to_analyze[goals_col] > gl).sum()
        op = round((oc / total_matches) * 100, 2) if total_matches > 0 else 0
        over_data.append([f"Over {gl}", oc, op, odd_min_from_percent(op)])
        uc = (df_to_analyze[goals_col] <= gl).sum()
        up = round((uc / total_matches) * 100, 2) if total_matches > 0 else 0
        under_data.append([f"Under {gl}", uc, up, odd_min_from_percent(up)])
    return (pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]),
            pd.DataFrame(under_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

def calcola_double_chance(df_to_analyze, period):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    df_double_chance = df_to_analyze.copy()
    if period == 'ft':
        df_double_chance["gol_home"] = df_double_chance["home_team_goal_count"]
        df_double_chance["gol_away"] = df_double_chance["away_team_goal_count"]
    elif period == 'ht':
        df_double_chance["gol_home"] = df_double_chance["home_team_goal_count_half_time"]
        df_double_chance["gol_away"] = df_double_chance["away_team_goal_count_half_time"]
    elif period == 'sh':
        df_double_chance["gol_home"] = df_double_chance["home_team_goal_count"] - df_double_chance["home_team_goal_count_half_time"]
        df_double_chance["gol_away"] = df_double_chance["away_team_goal_count"] - df_double_chance["away_team_goal_count_half_time"]
    else:
        st.error("Periodo non valido per il calcolo della doppia chance.")
        return pd.DataFrame()
    total_matches = len(df_double_chance)
    count_1x = ((df_double_chance["gol_home"] >= df_double_chance["gol_away"])).sum()
    count_x2 = ((df_double_chance["gol_away"] >= df_double_chance["gol_home"])).sum()
    count_12 = (df_double_chance["gol_home"] != df_double_chance["gol_away"]).sum()
    data = [
        ["1X", count_1x, round((count_1x / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["X2", count_x2, round((count_x2 / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["12", count_12, round((count_12 / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: odd_min_from_percent(x) if x > 0 else np.nan)
    return df_stats

def calcola_multi_gol(df_to_analyze, col_gol, titolo):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=[f"Mercato ({titolo})", "Conteggio", "Percentuale %", "Odd Minima"])
    total_matches = len(df_to_analyze)
    ranges = [
        ("0-1", lambda x: (x >= 0) & (x <= 1)),
        ("1-2", lambda x: (x >= 1) & (x <= 2)),
        ("2-3", lambda x: (x >= 2) & (x <= 3)),
        ("3+", lambda x: (x >= 3))
    ]
    data = []
    for label, cond in ranges:
        count = df_to_analyze[cond(df_to_analyze[col_gol])].shape[0]
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        data.append([f"Multi Gol {label}", count, perc, odd_min_from_percent(perc)])
    return pd.DataFrame(data, columns=[f"Mercato ({titolo})", "Conteggio", "Percentuale %", "Odd Minima"])

def calcola_first_to_score(df_to_analyze, period='ft'):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    risultati = {"Home Team": 0, "Away Team": 0, "No Goals": 0}
    total_matches = len(df_to_analyze)
    for _, row in df_to_analyze.iterrows():
        s_home = str(row.get("home_team_goal_timings", ""))
        s_away = str(row.get("away_team_goal_timings", ""))
        if period == 'ft':
            gh = _parse_goal_tokens(s_home, "base_minute"); ga = _parse_goal_tokens(s_away, "base_minute")
        elif period == 'ht':
            gh = [m for m in _parse_goal_tokens(s_home, "base_minute") if m <= 45]
            ga = [m for m in _parse_goal_tokens(s_away, "base_minute") if m <= 45]
        elif period == 'sh':
            gh = [m for m in _parse_goal_tokens(s_home, "base_minute") if m > 45]
            ga = [m for m in _parse_goal_tokens(s_away, "base_minute") if m > 45]
        else:
            continue
        minh = min(gh) if gh else float('inf')
        mina = min(ga) if ga else float('inf')
        if minh < mina: risultati["Home Team"] += 1
        elif mina < minh: risultati["Away Team"] += 1
        else:
            if minh == float('inf'): risultati["No Goals"] += 1
    stats = []
    for esito, cnt in risultati.items():
        perc = round((cnt / total_matches) * 100, 2) if total_matches > 0 else 0
        stats.append((esito, cnt, perc, odd_min_from_percent(perc)))
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

def calcola_btts(df_to_analyze, period):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    df_btts = df_to_analyze.copy()
    if period == 'ht':
        btts_count = ((df_btts["home_team_goal_count_half_time"] > 0) & (df_btts["away_team_goal_count_half_time"] > 0)).sum()
    elif period == 'ft':
        btts_count = ((df_btts["home_team_goal_count"] > 0) & (df_btts["away_team_goal_count"] > 0)).sum()
    elif period == 'sh':
        df_btts["sh_home_goals"] = df_btts["home_team_goal_count"] - df_btts["home_team_goal_count_half_time"]
        df_btts["sh_away_goals"] = df_btts["away_team_goal_count"] - df_btts["away_team_goal_count_half_time"]
        btts_count = ((df_btts["sh_home_goals"] > 0) & (df_btts["sh_away_goals"] > 0)).sum()
    else:
        return pd.DataFrame()
    total_matches = len(df_btts)
    no_btts_count = total_matches - btts_count
    data = [
        [f"BTTS SI ({period.upper()})", btts_count, round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        [f"BTTS NO ({period.upper()})", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(odd_min_from_percent)
    return df_stats

def calcola_to_score(df_to_analyze, period):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_to_score = df_to_analyze.copy()
    if period == 'ht':
        home_to_score_count = (df_to_score["home_team_goal_count_half_time"] > 0).sum()
        away_to_score_count = (df_to_score["away_team_goal_count_half_time"] > 0).sum()
    elif period == 'sh':
        df_to_score["sh_home_goals"] = df_to_score["home_team_goal_count"] - df_to_score["home_team_goal_count_half_time"]
        df_to_score["sh_away_goals"] = df_to_score["away_team_goal_count"] - df_to_score["away_team_goal_count_half_time"]
        home_to_score_count = (df_to_score["sh_home_goals"] > 0).sum()
        away_to_score_count = (df_to_score["sh_away_goals"] > 0).sum()
    else:  # ft
        home_to_score_count = (df_to_score["home_team_goal_count"] > 0).sum()
        away_to_score_count = (df_to_score["away_team_goal_count"] > 0).sum()
    total_matches = len(df_to_score)
    if total_matches == 0:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    data = [
        [f"Home Team to Score ({period.upper()})", home_to_score_count, round((home_to_score_count / total_matches) * 100, 2)],
        [f"Away Team to Score ({period.upper()})", away_to_score_count, round((away_to_score_count / total_matches) * 100, 2)]
    ]
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(odd_min_from_percent)
    return df_stats

def calcola_next_goal(df_to_analyze, start_min, end_min):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    risultati = {"Prossimo Gol: Home": 0, "Prossimo Gol: Away": 0, "Nessun prossimo gol": 0}
    total_matches = len(df_to_analyze)
    for _, row in df_to_analyze.iterrows():
        gol_home = get_goal_minutes(row, "home", mapping="base_minute")
        gol_away = get_goal_minutes(row, "away", mapping="base_minute")
        next_home_goal = min([g for g in gol_home if start_min <= g <= end_min] or [float('inf')])
        next_away_goal = min([g for g in gol_away if start_min <= g <= end_min] or [float('inf')])
        if next_home_goal < next_away_goal:
            risultati["Prossimo Gol: Home"] += 1
        elif next_away_goal < next_home_goal:
            risultati["Prossimo Gol: Away"] += 1
        else:
            if next_home_goal == float('inf'):
                risultati["Nessun prossimo gol"] += 1
    stats = []
    for esito, count in risultati.items():
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        stats.append((esito, count, perc, odd_min_from_percent(perc)))
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# =======================
# Visualizzazioni (Timeband con 45+ e 90+)
# =======================
def mostra_distribuzione_timeband(df_to_analyze, min_start_display=0):
    """
    Timeband con bucket dedicati per 45+ e 90+.
    - Intervalli regolari: 0-15, 16-30, 31-45, 46-60, 61-75, 76-90 (bordi inclusivi)
    - Speciali: 45+ e 90+ conteggiati separatamente (solo se min_start_display <= 45 / <= 90)
    Nota: in tutti gli altri calcoli fuori da questa funzione 45+ è trattato come 45 (base_minute).
    """
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 15 minuti è vuoto.")
        return

    bands = [
        ((0, 15),  "0-15"),
        ((16, 30), "16-30"),
        ((31, 45), "31-45"),
        ((46, 60), "46-60"),
        ((61, 75), "61-75"),
        ((76, 90), "76-90"),
    ]

    risultati = []
    total_matches = len(df_to_analyze)

    # Contatori speciali 45+ e 90+
    cnt_45p_matches = cnt_45p_matches_2plus = 0
    cnt_90p_matches = cnt_90p_matches_2plus = 0

    for ((start_interval, end_interval), label) in bands:
        if end_interval < min_start_display:
            continue

        partite_con_gol = 0
        partite_con_almeno_2_gol = 0

        for _, row in df_to_analyze.iterrows():
            s_home = str(row.get("home_team_goal_timings", ""))
            s_away = str(row.get("away_team_goal_timings", ""))

            tok_home = GOAL_TOKEN_RE.findall(s_home)
            tok_away = GOAL_TOKEN_RE.findall(s_away)

            reg_home, reg_away = [], []
            h_45p = h_90p = a_45p = a_90p = 0

            for tok in tok_home:
                if '+' in tok:
                    base, extra = tok.split('+'); base = int(base)
                    if base == 45: h_45p += 1
                    elif base == 90: h_90p += 1
                    else: reg_home.append(base)
                else:
                    reg_home.append(int(tok))

            for tok in tok_away:
                if '+' in tok:
                    base, extra = tok.split('+'); base = int(base)
                    if base == 45: a_45p += 1
                    elif base == 90: a_90p += 1
                    else: reg_away.append(base)
                else:
                    reg_away.append(int(tok))

            lb = max(start_interval, min_start_display)
            goals_in_interval_home = [m for m in reg_home if lb <= m <= end_interval]
            goals_in_interval_away = [m for m in reg_away if lb <= m <= end_interval]
            total_goals_in_interval = len(goals_in_interval_home) + len(goals_in_interval_away)

            if total_goals_in_interval > 0:
                partite_con_gol += 1
            if total_goals_in_interval >= 2:
                partite_con_almeno_2_gol += 1

            if min_start_display <= 45:
                tot_45p = h_45p + a_45p
                if tot_45p > 0: cnt_45p_matches += 1
                if tot_45p >= 2: cnt_45p_matches_2plus += 1

            if min_start_display <= 90:
                tot_90p = h_90p + a_90p
                if tot_90p > 0: cnt_90p_matches += 1
                if tot_90p >= 2: cnt_90p_matches_2plus += 1

        perc_con_gol = round((partite_con_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min_con_gol = odd_min_from_percent(perc_con_gol)
        perc_almeno_2_gol = round((partite_con_almeno_2_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min_almeno_2_gol = odd_min_from_percent(perc_almeno_2_gol)

        risultati.append([label, partite_con_gol, perc_con_gol, odd_min_con_gol, perc_almeno_2_gol, odd_min_almeno_2_gol])

    # Inserisci 45+ subito dopo 31-45 (se visibile per lo start)
    if min_start_display <= 45:
        perc_45p = round((cnt_45p_matches / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_45p = odd_min_from_percent(perc_45p)
        perc_45p_2 = round((cnt_45p_matches_2plus / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_45p_2 = odd_min_from_percent(perc_45p_2)
        idx_3145 = next((i for i, r in enumerate(risultati) if r[0] == "31-45"), None)
        row_45p = ["45+", cnt_45p_matches, perc_45p, odd_45p, perc_45p_2, odd_45p_2]
        if idx_3145 is not None: risultati.insert(idx_3145 + 1, row_45p)
        else: risultati.append(row_45p)

    # Aggiungi 90+ in coda (se visibile per lo start)
    if min_start_display <= 90:
        perc_90p = round((cnt_90p_matches / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_90p = odd_min_from_percent(perc_90p)
        perc_90p_2 = round((cnt_90p_matches_2plus / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_90p_2 = odd_min_from_percent(perc_90p_2)
        risultati.append(["90+", cnt_90p_matches, perc_90p, odd_90p, perc_90p_2, odd_90p_2])

    if not risultati:
        st.info(f"Nessun intervallo di tempo rilevante dopo il minuto {min_start_display}.")
        return

    df_result = pd.DataFrame(risultati, columns=[
        "Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima", ">= 2 Gol %", "Odd Minima >= 2 Gol"
    ])
    

# ===================== STATISTICHE SU df_out =====================
def _pct(n, d):
    return (100.0 * n / d) if d else 0.0

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

# Derivazioni sicure per FT e HT
        # --- GUARD: ensure df_out exists ---
        if 'df_out' not in locals() and 'df_out' not in globals():
            try:
                df_out = df_after_second.copy() if 'df_after_second' in locals() or 'df_after_second' in globals() else pd.DataFrame()
            except Exception:
                df_out = pd.DataFrame()
        # If still undefined or None, make it a DataFrame
        if df_out is None:
            df_out = pd.DataFrame()
tot_rows = len(df_out)

# FT goal counts
if {'home_team_goal_count','away_team_goal_count'}.issubset(df_out.columns):
    ft_h = df_out['home_team_goal_count'].astype('Int64').fillna(0).astype(int)
    ft_a = df_out['away_team_goal_count'].astype('Int64').fillna(0).astype(int)
else:
    # Fallback: conta da timings
    def _parse_list(s):
        if s is None: return []
        s = str(s)
        if s.strip() == '' or s.lower() == 'nan': return []
        parts = [p.strip() for p in s.split(',') if p.strip()!='']
        out = []
        for p in parts:
            try:
                out.append(int(p.split('+')[0].split("'")[0]))
            except Exception:
                pass
        return out
    ft_h = df_out.get('home_team_goal_timings', []).apply(_parse_list).apply(len) if 'home_team_goal_timings' in df_out.columns else 0
    ft_a = df_out.get('away_team_goal_timings', []).apply(_parse_list).apply(len) if 'away_team_goal_timings' in df_out.columns else 0

# HT goal counts
if {'home_team_goal_count_half_time','away_team_goal_count_half_time'}.issubset(df_out.columns):
    ht_h = df_out['home_team_goal_count_half_time'].astype('Int64').fillna(0).astype(int)
    ht_a = df_out['away_team_goal_count_half_time'].astype('Int64').fillna(0).astype(int)
else:
    # Fallback da timings (<=45 inclusivo)
    def _count_ht(s):
        if s is None: return 0
        s = str(s)
        if s.strip()=='' or s.lower()=='nan': return 0
        cnt = 0
        for p in s.split(','):
            p = p.strip()
            if not p: continue
            try:
                m = int(p.split('+')[0].split("'")[0])
                if m <= 45: cnt += 1
            except Exception:
                pass
        return cnt
    ht_h = df_out.get('home_team_goal_timings', []).apply(_count_ht) if 'home_team_goal_timings' in df_out.columns else 0
    ht_a = df_out.get('away_team_goal_timings', []).apply(_count_ht) if 'away_team_goal_timings' in df_out.columns else 0

# SH = FT - HT
sh_h = (ft_h - ht_h)
sh_a = (ft_a - ht_a)

# Totali
ft_tot = ft_h + ft_a
ht_tot = ht_h + ht_a
sh_tot = sh_h + sh_a

# ---------- Over Winrate ----------
def _over_rate(series, th):
    n = int((series > th).sum())
    return n, _pct(n, tot_rows)

over_ft = {}
for th in [0,1,2,3,4]:  # Over 0.5 .. Over 4.5
    n, r = _over_rate(ft_tot, th)
    over_ft[f"Over {th+0.5} FT"] = {"N": n, "Win%": round(r,2)}

over_ht = {}
for th in [0,1]:  # Over 0.5/1.5 HT
    n, r = _over_rate(ht_tot, th)
    over_ht[f"Over {th+0.5} HT"] = {"N": n, "Win%": round(r,2)}

over_sh = {}
for th in [0,1,2]:  # Over 0.5/1.5/2.5 SH
    n, r = _over_rate(sh_tot, th)
    over_sh[f"Over {th+0.5} 2°T"] = {"N": n, "Win%": round(r,2)}

# ---------- Esiti esatti ----------
from collections import Counter
exact_ft = Counter(zip(ft_h, ft_a))
exact_ht = Counter(zip(ht_h, ht_a))
exact_sh = Counter(zip(sh_h, sh_a))

def _top_scores(counter, topk=10):
    items = counter.most_common(topk)
    return [{"Score": f"{h}-{a}", "N": n, "Win%": round(_pct(n, tot_rows),2)} for (h,a), n in items]

# ---------- 1X2 Winrate ----------
# FT
def _sign(x): return (0 if x==0 else (1 if x>0 else -1))
ft_sign = (ft_h - ft_a).apply(_sign)
n_home = int((ft_sign > 0).sum())
n_draw = int((ft_sign == 0).sum())
n_away = int((ft_sign < 0).sum())
res_1x2_ft = [
    {"Esito": "1", "N": n_home, "Win%": round(_pct(n_home, tot_rows),2)},
    {"Esito": "X", "N": n_draw, "Win%": round(_pct(n_draw, tot_rows),2)},
    {"Esito": "2", "N": n_away, "Win%": round(_pct(n_away, tot_rows),2)},
]

# HT
ht_sign = (ht_h - ht_a).apply(_sign)
hth = int((ht_sign > 0).sum())
htx = int((ht_sign == 0).sum())
hta = int((ht_sign < 0).sum())
res_1x2_ht = [
    {"Esito": "1", "N": hth, "Win%": round(_pct(hth, tot_rows),2)},
    {"Esito": "X", "N": htx, "Win%": round(_pct(htx, tot_rows),2)},
    {"Esito": "2", "N": hta, "Win%": round(_pct(hta, tot_rows),2)},
]

# SH (solo differenziale 2°T)
sh_sign = (sh_h - sh_a).apply(_sign)
shh = int((sh_sign > 0).sum())
shx = int((sh_sign == 0).sum())
sha = int((sh_sign < 0).sum())
res_1x2_sh = [
    {"Esito": "1", "N": shh, "Win%": round(_pct(shh, tot_rows),2)},
    {"Esito": "X", "N": shx, "Win%": round(_pct(shx, tot_rows),2)},
    {"Esito": "2", "N": sha, "Win%": round(_pct(sha, tot_rows),2)},
]

import pandas as _pd
st.markdown("### 📊 Statistiche dal sottoinsieme corrente")
# Toggles per mostrare/nascondere le sezioni
col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    _show_ft = st.checkbox("Mostra FT", value=True, key="show_ft_stats")
with col_t2:
    _show_ht = st.checkbox("Mostra HT", value=True, key="show_ht_stats")
with col_t3:
    _show_sh = st.checkbox("Mostra 2°T (SH)", value=False, key="show_sh_stats")

with st.container():
    if _show_ft:
        with st.expander("FT — Over winrate & 1X2", expanded=False):
            st.write(_pd.DataFrame(over_ft).T)
            st.write(_pd.DataFrame(res_1x2_ft))
            st.markdown("**Top 10 risultati esatti FT**")
            st.write(_pd.DataFrame(_top_scores(exact_ft, 10)))

    if _show_ht:
        with st.expander("HT — Over winrate & 1X2", expanded=False):
            st.write(_pd.DataFrame(over_ht).T)
            st.write(_pd.DataFrame(res_1x2_ht))
            st.markdown("**Top 10 risultati esatti HT**")
            st.write(_pd.DataFrame(_top_scores(exact_ht, 10)))

    if _show_sh:
        with st.expander("2° Tempo — Over winrate & 1X2", expanded=False):
            st.write(_pd.DataFrame(over_sh).T)
            st.write(_pd.DataFrame(res_1x2_sh))
            st.markdown("**Top 10 risultati esatti 2° Tempo**")
            st.write(_pd.DataFrame(_top_scores(exact_sh, 10)))
# =================== FINE STATISTICHE ===================


st.dataframe(style_table(df_result, ['Percentuale %', '>= 2 Gol %']))

def mostra_risultati_esatti(df, col_home, col_away, titolo):
    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]
    df_valid = df[(df[col_home].notna()) & (df[col_away].notna())].copy()
    if df_valid.empty:
        st.subheader(f"Risultati Esatti {titolo} (0 partite)")
        st.info("Nessun dato valido per i risultati esatti nel dataset filtrato.")
        return

    def classifica_risultato(row):
        try:
            home, away = int(row[col_home]), int(row[col_away])
        except:
            return "Altro"
        ris = f"{home}-{away}"
        if ris in risultati_interessanti:
            return ris
        if home > away: return "Altro risultato casa vince"
        elif home < away: return "Altro risultato ospite vince"
        else: return "Altro pareggio"

    df_valid["classificato"] = df_valid.apply(classifica_risultato, axis=1)
    distribuzione = df_valid["classificato"].value_counts().reset_index()
    distribuzione.columns = [titolo, "Conteggio"]
    distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df_valid) * 100).round(2)
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(odd_min_from_percent)
    st.subheader(f"Risultati Esatti {titolo} ({len(df_valid)} partite)")
    st.dataframe(style_table(distribuzione, ['Percentuale %']))

# =======================
# Caricamento dati
# =======================
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=';', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Errore durante la lettura del file. Assicurati che sia un file CSV con separatore ';'. Errore: {e}")
        return pd.DataFrame()

    num_cols = [
        'home_team_goal_count_half_time','away_team_goal_count_half_time',
        'home_team_goal_count','away_team_goal_count',
        'home_team_shots_on_target','away_team_shots_on_target',
        'odds_ft_home_team_win','odds_ft_draw','odds_ft_away_team_win',
        'odds_ft_over25','anno','Game Week'
    ]
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col].replace({',':'.'}, regex=True, inplace=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'status' in df.columns:
        df = df[df['status'].str.lower() != 'incomplete']

    if {'giorno', 'mese', 'anno'}.issubset(df.columns) and 'date' not in df.columns:
        df['date'] = pd.to_datetime(df[['giorno', 'mese', 'anno']].astype(str).agg('-'.join, axis=1), format='%d-%m-%Y', errors='coerce')

    if {'home_team_goal_count_half_time','away_team_goal_count_half_time'}.issubset(df.columns):
        def get_ht_result(row):
            if row['home_team_goal_count_half_time'] > row['away_team_goal_count_half_time']:
                return 'Vittoria Casa'
            elif row['home_team_goal_count_half_time'] < row['away_team_goal_count_half_time']:
                return 'Vittoria Trasferta'
            else:
                return 'Pareggio'
        df['Risultato HT'] = df.apply(get_ht_result, axis=1)
        df['HT Score'] = df['home_team_goal_count_half_time'].astype('Int64').astype(str) + ' - ' + df['away_team_goal_count_half_time'].astype('Int64').astype(str)
        df['total_goals_at_half_time'] = df['home_team_goal_count_half_time'] + df['away_team_goal_count_half_time']

    if {'home_team_goal_count','away_team_goal_count'}.issubset(df.columns):
        df['total_goals_at_full_time'] = df['home_team_goal_count'] + df['away_team_goal_count']
    
    if 'date' in df.columns and not df['date'].isnull().all():
        df.sort_values(by='date', ascending=True, inplace=True)
    else:
        st.sidebar.warning("Colonna 'date' non valida. Le partite non verranno ordinate cronologicamente.")

    return df

# =======================
# UI
# =======================
st.set_page_config(page_title="Filtri Dati Calcio", layout="wide")
st.title("⚽ Dashboard Pattern Gol Calcio")
st.write("Carica il tuo file CSV per iniziare l'analisi.")
uploaded_file = st.file_uploader("Scegli un file CSV", type=["csv"], key="uploader1")

if uploaded_file is None:
    st.info("In attesa di caricamento del file CSV.")
    st.stop()

with st.spinner('Caricamento dati in corso...'):
    df = load_data(uploaded_file)

if df.empty:
    st.error("Il file caricato è vuoto o non può essere processato.")
    st.stop()

st.success("File caricato con successo!")

# --- UI FILTERS ---
st.sidebar.header("Opzioni di Filtraggio")
selected_leagues = []
if 'league' in df.columns:
    leagues = sorted(df['league'].dropna().unique().tolist())
    selected_leagues = st.sidebar.multiselect("Seleziona Campionato(i)", leagues, default=[])
else:
    st.sidebar.warning("Colonna 'league' non trovata.")

selected_years = []
if 'anno' in df.columns and not df['anno'].isnull().all():
    years_series = df['anno'].dropna().astype(int)
    max_year = int(years_series.max())
    year_options = {
        'Ultimo anno': [max_year],
        'Ultimi 2 anni': list(range(max_year-1, max_year+1)),
        'Ultimi 3 anni': list(range(max_year-2, max_year+1)),
        'Ultimi 4 anni': list(range(max_year-3, max_year+1)),
        'Ultimi 5 anni': list(range(max_year-4, max_year+1)),
        'Ultimi 6 anni': list(range(max_year-5, max_year+1)),
        'Ultimi 7 anni': list(range(max_year-6, max_year+1)),
        'Ultimi 8 anni': list(range(max_year-7, max_year+1)),
        'Ultimi 9 anni': list(range(max_year-8, max_year+1)),
        'Ultimi 10 anni': list(range(max_year-9, max_year+1)),
        'Tutti': sorted(years_series.unique().tolist())
    }
    year_choice = st.sidebar.selectbox("Seleziona Anno/i", options=list(year_options.keys()), index=len(year_options) - 1)
    selected_years = year_options[year_choice]
else:
    st.sidebar.warning("Colonna 'anno' non trovata.")

selected_gw = None
if 'Game Week' in df.columns and not df['Game Week'].isnull().all():
    gws = sorted(df['Game Week'].dropna().astype(int).unique().tolist())
    gw_min, gw_max = min(gws), max(gws)
    selected_gw = st.sidebar.slider("Range Giornata", min_value=gw_min, max_value=gw_max, value=(gw_min, gw_max))
else:
    st.sidebar.warning("Colonna 'Game Week' non trovata.")

teams = []
if 'home_team_name' in df.columns and 'away_team_name' in df.columns:
    all_teams = pd.unique(df[['home_team_name', 'away_team_name']].values.ravel('K'))
    teams = sorted(all_teams.tolist())
    selected_home_team = st.sidebar.selectbox("Seleziona Squadra di Casa", ['Tutte'] + teams, index=0)
    selected_away_team = st.sidebar.selectbox("Seleziona Squadra in Trasferta", ['Tutte'] + teams, index=0)
else:
    selected_home_team = 'Tutte'
    selected_away_team = 'Tutte'
    st.sidebar.warning("Colonne 'home_team_name' e/o 'away_team_name' non trovate.")

last_matches_count = 'Tutte'
total_analysis_toggle = False
if (selected_home_team != 'Tutte' or selected_away_team != 'Tutte') and ('date' in df.columns or ('anno' in df.columns and 'Game Week' in df.columns)):
    num_matches_options = ['Tutte'] + [3, 5, 10, 15, 20, 25, 30, 50, 60, 75, 90, 100]
    last_matches_count = st.sidebar.selectbox("Analizza le ultime N partite", options=num_matches_options)
    total_analysis_toggle = st.sidebar.checkbox("Analizza Partite Totali (Home/Away)", value=False)
else:
    st.sidebar.warning("Colonne necessarie non presenti per l'analisi squadra per squadra.")

selected_ht_results = []
if 'HT Score' in df.columns:
    ht_scores = sorted(df['HT Score'].dropna().unique().tolist())
    selected_ht_results = st.sidebar.multiselect("Filtra Risultato HT", ht_scores, default=[])
else:
    st.sidebar.warning("Colonna 'HT Score' non trovata.")

st.sidebar.subheader("Filtri Quote FT (opzionali)")
odds_filters = {}
for label, col in {'Casa':'odds_ft_home_team_win','X':'odds_ft_draw','Trasferta':'odds_ft_away_team_win'}.items():
    if col in df.columns:
        mn = st.sidebar.number_input(f"Min {label}", min_value=1.01, step=0.01, key=f"min_{col}")
        mx = st.sidebar.number_input(f"Max {label}", min_value=1.01, step=0.01, value=100.00, key=f"max_{col}")
        odds_filters[col] = (mn, mx)

# --- APPLY FILTERS ---
base_filtered = df.copy()
if selected_leagues: base_filtered = base_filtered[base_filtered['league'].isin(selected_leagues)]
if selected_years: base_filtered = base_filtered[base_filtered['anno'].isin(selected_years)]
if selected_gw: base_filtered = base_filtered[(base_filtered['Game Week'] >= selected_gw[0]) & (base_filtered['Game Week'] <= selected_gw[1])]
if selected_ht_results and 'HT Score' in base_filtered.columns:
    base_filtered = base_filtered[base_filtered['HT Score'].isin(selected_ht_results)]

team_filtered_df = pd.DataFrame()
if selected_home_team != 'Tutte' or selected_away_team != 'Tutte':
    if 'home_team_name' in base_filtered.columns and 'away_team_name' in base_filtered.columns:
        if total_analysis_toggle and last_matches_count != 'Tutte':
            team_df_home = base_filtered[
                (base_filtered['home_team_name'] == selected_home_team) |
                (base_filtered['away_team_name'] == selected_home_team)
            ].copy()
            team_df_away = base_filtered[
                (base_filtered['home_team_name'] == selected_away_team) |
                (base_filtered['away_team_name'] == selected_away_team)
            ].copy()
            sort_cols = ['anno', 'Game Week']
            if 'date' in team_df_home.columns and not team_df_home['date'].isnull().all():
                sort_cols = ['date']
            home_last_n = team_df_home.sort_values(by=sort_cols, ascending=False).head(last_matches_count)
            away_last_n = team_df_away.sort_values(by=sort_cols, ascending=False).head(last_matches_count)
            team_filtered_df = pd.concat([home_last_n, away_last_n]).drop_duplicates().sort_values(by=sort_cols, ascending=False)
        else:
            if selected_home_team != 'Tutte' and selected_away_team != 'Tutte':
                home_df = base_filtered[base_filtered['home_team_name'] == selected_home_team].copy()
                away_df = base_filtered[base_filtered['away_team_name'] == selected_away_team].copy()
                if last_matches_count != 'Tutte':
                    sort_cols = ['anno', 'Game Week']
                    if 'date' in home_df.columns and not home_df['date'].isnull().all():
                        sort_cols = ['date']
                    home_df = home_df.sort_values(by=sort_cols, ascending=False).head(last_matches_count)
                    away_df = away_df.sort_values(by=sort_cols, ascending=False).head(last_matches_count)
                team_filtered_df = pd.concat([home_df, away_df])
            elif selected_home_team != 'Tutte':
                team_filtered_df = base_filtered[base_filtered['home_team_name'] == selected_home_team]
                if last_matches_count != 'Tutte':
                    if 'date' in team_filtered_df.columns and not team_filtered_df['date'].isnull().all():
                        team_filtered_df = team_filtered_df.sort_values(by='date', ascending=False).head(last_matches_count)
                    else:
                        team_filtered_df = team_filtered_df.sort_values(by=['anno', 'Game Week'], ascending=False).head(last_matches_count)
            elif selected_away_team != 'Tutte':
                team_filtered_df = base_filtered[base_filtered['away_team_name'] == selected_away_team]
                if last_matches_count != 'Tutte':
                    if 'date' in team_filtered_df.columns and not team_filtered_df['date'].isnull().all():
                        team_filtered_df = team_filtered_df.sort_values(by='date', ascending=False).head(last_matches_count)
                    else:
                        team_filtered_df = team_filtered_df.sort_values(by=['anno', 'Game Week'], ascending=False).head(last_matches_count)
    else:
        team_filtered_df = base_filtered.copy()
else:
    team_filtered_df = base_filtered.copy()

odds_filtered = team_filtered_df.copy()
for col, (mn, mx) in odds_filters.items():
    if col in odds_filtered.columns:
        odds_filtered = odds_filtered[(odds_filtered[col] >= mn) & (odds_filtered[col] <= mx)]

# --- MAIN SECTION: Analisi Pattern Gol ---
st.subheader("Analisi Pattern Gol")
st.write("Analizza le partite in base a specifici pattern di gol.")

goal_pattern_time_intervals = {
    "0-5": (0, 5), "0-10": (0, 10), "11-20": (11, 20), "21-30": (21, 30),
    "31-39": (31, 39), "40-45": (40, 45), "46-55": (46, 55), "56-65": (56, 65),
    "66-75": (66, 75), "75-80": (75, 80), "75-90": (75, 90), "80-90": (80, 90),
    "85-90": (85, 90)
}
time_interval_options = ["Nessun Filtro"] + list(goal_pattern_time_intervals.keys())

st.markdown("### Primo Gol (opzionale)")
col1_patt, col2_patt = st.columns(2)
with col1_patt:
    first_goal_result = st.selectbox("Risultato dopo il primo gol", ["Nessun Filtro", "1-0", "0-1"], key="first_goal_res")
with col2_patt:
    first_goal_time = st.selectbox("Intervallo di tempo primo gol", time_interval_options, key="first_goal_time")

if first_goal_result != "Nessun Filtro":
    st.markdown("### Secondo Gol (opzionale)")
    col3_patt, col4_patt = st.columns(2)
    with col3_patt:
        second_goal_result = st.selectbox("Risultato dopo il secondo gol", ["Nessun Filtro", "2-0", "1-1", "0-2"], key="second_goal_res")
    with col4_patt:
        second_goal_time = st.selectbox("Intervallo di tempo secondo gol", time_interval_options, key="second_goal_time")
else:
    second_goal_result = "Nessun Filtro"
    second_goal_time = "Nessun Filtro"

st.markdown("### Filtri Quote (per questa analisi)")
col_patt_quote1, col_patt_quote2 = st.columns(2)
with col_patt_quote1:
    odd_home_min_patt = st.text_input("Odd Home Min", value="1.0", key="odd_home_patt_min")
    odd_home_max_patt = st.text_input("Odd Home Max", value="20.0", key="odd_home_patt_max")
with col_patt_quote2:
    odd_away_min_patt = st.text_input("Odd Away Min", value="1.0", key="odd_away_patt_min")
    odd_away_max_patt = st.text_input("Odd Away Max", value="20.0", key="odd_away_patt_max")
    
start_min_patt = st.slider("Minuto iniziale (vincolo 1)", 1, 90, 1, key="start_min_patt")
risultato_attuale_patt = st.text_input("Risultato al minuto (vincolo 1) es. '1-0'", value="0-0", key="risultato_attuale_patt")

use_second_constraint = st.checkbox("Aggiungi secondo vincolo (minuto + risultato)", value=False)
if use_second_constraint:
    second_min_patt = st.slider("Secondo minuto (vincolo 2)", 1, 90, max(start_min_patt, 46), key="second_min_patt")
    risultato_attuale_patt2 = st.text_input("Risultato al secondo minuto (vincolo 2) es. '1-0'", value=risultato_attuale_patt, key="risultato_attuale_patt2")
else:
    second_min_patt = None
    risultato_attuale_patt2 = None


if st.button("Avvia Analisi Pattern Gol"):
    try:
        df_pattern = odds_filtered.copy() if 'odds_filtered' in globals() else df.copy() if 'df' in globals() else pd.DataFrame()

        # Safe defaults for odds filters
        try:
            _ohmin = str(odd_home_min_patt).strip() if 'odd_home_min_patt' in globals() else ""
            _ohmax = str(odd_home_max_patt).strip() if 'odd_home_max_patt' in globals() else ""
            _oamin = str(odd_away_min_patt).strip() if 'odd_away_min_patt' in globals() else ""
            _oamax = str(odd_away_max_patt).strip() if 'odd_away_max_patt' in globals() else ""
            odd_home_min = float(_ohmin) if _ohmin else 1.0
            odd_home_max = float(_ohmax) if _ohmax else 20.0
            odd_away_min = float(_oamin) if _oamin else 1.0
            odd_away_max = float(_oamax) if _oamax else 20.0
        except Exception:
            st.error("I valori delle quote per l'analisi Pattern non sono validi. Inserisci numeri.")
            df_pattern = pd.DataFrame()

        if not df_pattern.empty and 'odds_ft_home_team_win' in df_pattern.columns:
            df_pattern = df_pattern[(df_pattern['odds_ft_home_team_win'] >= odd_home_min) & (df_pattern['odds_ft_home_team_win'] <= odd_home_max)]
        if not df_pattern.empty and 'odds_ft_away_team_win' in df_pattern.columns:
            df_pattern = df_pattern[(df_pattern['odds_ft_away_team_win'] >= odd_away_min) & (df_pattern['odds_ft_away_team_win'] <= odd_away_max)]

        # Gate 1: risultato attuale (fa fede)
        try:
            curr_h, curr_a = map(int, str(risultato_attuale_patt).split('-'))
        except Exception:
            st.error("Formato risultato (vincolo 1) non valido. Usa 'X-Y' (es. '1-0').")
            st.stop()

        def _safe_int(x, d):
            try:
                return int(x)
            except Exception:
                return d

        _start_min = _safe_int(start_min_patt if 'start_min_patt' in globals() else 46, 46)

        def match_score_at_min(row):
            h_at_m, a_at_m = cum_score_at_min(row, _start_min)
            return (h_at_m == curr_h) and (a_at_m == curr_a)

        df_gate1 = df_pattern[df_pattern.apply(match_score_at_min, axis=1)]

        # PATCH A/B: Coerenza HT
        try:
            if _start_min >= 46 and {'home_team_goal_count_half_time','away_team_goal_count_half_time'}.issubset(df_gate1.columns):
                df_gate1 = df_gate1[(df_gate1['home_team_goal_count_half_time'].astype(int) == int(curr_h)) & (df_gate1['away_team_goal_count_half_time'].astype(int) == int(curr_a))]
        except Exception:
            pass
        try:
            if _start_min <= 45 and {'home_team_goal_count_half_time','away_team_goal_count_half_time'}.issubset(df_gate1.columns):
                df_gate1 = df_gate1[(df_gate1['home_team_goal_count_half_time'].astype(int) == int(curr_h)) & (df_gate1['away_team_goal_count_half_time'].astype(int) == int(curr_a))]
        except Exception:
            pass

        # Gate 2: se c'è secondo vincolo (minuto + risultato)
        df_gate2 = df_gate1
        if 'use_second_constraint' in globals() and use_second_constraint and second_min_patt is not None and risultato_attuale_patt2:
            try:
                curr2_h, curr2_a = map(int, str(risultato_attuale_patt2).split('-'))
                _second_min = _safe_int(second_min_patt, _start_min)
                def match_score_at_min2(row):
                    h2, a2 = cum_score_at_min(row, _second_min)
                    return (h2 == curr2_h) and (a2 == curr2_a)
                df_gate2 = df_gate2[df_gate2.apply(match_score_at_min2, axis=1)]
                # Se si attraversa l'intervallo senza cambio punteggio, imponi zero gol fino a HT
                if _start_min <= 45 and _second_min >= 46 and 'home_team_goal_count_half_time' in df_gate2.columns:
                    def no_goals_until_ht_and_ht_equals(row):
                        g_h, g_a = goals_between(row, _start_min, 45)
                        ht_ok = (int(row['home_team_goal_count_half_time']) == curr_h) and (int(row['away_team_goal_count_half_time']) == curr_a)
                        return (g_h == 0 and g_a == 0 and ht_ok)
                    df_gate2 = df_gate2[df_gate2.apply(no_goals_until_ht_and_ht_equals, axis=1)]
            except Exception:
                pass

        # Sezione filtri 1°/2° gol
        df_after_first = df_gate2.copy()
        try:
            if 'first_goal_result' in globals() and 'first_goal_time' in globals() and first_goal_result != "Nessun Filtro" and first_goal_time != "Nessun Filtro":
                fh, fa = map(int, str(first_goal_result).split('-'))
                min_first, max_first = goal_pattern_time_intervals.get(str(first_goal_time), (None, None)) if 'goal_pattern_time_intervals' in globals() else (None, None)
                if min_first is not None and max_first is not None:
                    df_after_first = df_after_first[df_after_first.apply(lambda row: check_first_goal_enhanced(row, fh, fa, min_first, max_first), axis=1)]
        except Exception:
            pass

        # Secondo gol (robusto)
        try:
            _apply_second = False
            if 'second_goal_result' in globals() and 'second_goal_time' in globals() and second_goal_result != "Nessun Filtro" and second_goal_time != "Nessun Filtro":
                sh, sa = map(int, str(second_goal_result).split('-'))
                intervals = goal_pattern_time_intervals if 'goal_pattern_time_intervals' in globals() else {"0-10": (0,10), "11-20": (11,20), "21-30": (21,30), "31-39": (31,39), "40-45": (40,45), "45+": (45,45), "46-60": (46,60), "61-75": (61,75), "76-90": (76,90), "90+": (90,90)}
                min_second, max_second = intervals.get(str(second_goal_time), (None, None))
                if min_second is not None and max_second is not None:
                    _apply_second = True
            if _apply_second:
                df_after_second = df_after_first[df_after_first.apply(lambda row: check_second_goal_enhanced(row, sh, sa, min_second, max_second), axis=1)]
            else:
                df_after_second = df_after_first.copy()
        except Exception:
            df_after_second = df_after_first.copy()

        # Sigillo HT se 2° gol nel 1° tempo
        try:
            if _apply_second and max_second is not None and max_second <= 45 and {'home_team_goal_count_half_time','away_team_goal_count_half_time'}.issubset(df_after_second.columns):
                df_after_second = df_after_second[
                    (df_after_second['home_team_goal_count_half_time'].astype(int) == sh) &
                    (df_after_second['away_team_goal_count_half_time'].astype(int) == sa)
                ]
        except Exception:
            pass

        df_out = df_after_second.copy()

        st.markdown("---")
        if df_out.empty:
            st.warning("Nessuna partita trovata con i vincoli e i filtri selezionati.")
        else:
            st.success(f"Partite trovate: {len(df_out)}")
            st.dataframe(df_out[['date_GMT','home_team_name','away_team_name','home_team_goal_timings','away_team_goal_timings','total_goals_at_half_time']].head(50) if set(['date_GMT','home_team_name','away_team_name','home_team_goal_timings','away_team_goal_timings','total_goals_at_half_time']).issubset(df_out.columns) else df_out.head(50))
    except Exception as e:
        st.error(f"Errore durante l'analisi pattern: {e}")
        import traceback as _tb
        st.code(_tb.format_exc())
