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
# Funzioni pattern PRIMO/SECONDO gol (senza controlli HT/FT; la coerenza è gestita dai vincoli)
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
            if home > away:
                risultati["1 (Casa)"] += 1
            elif home < away:
                risultati["2 (Trasferta)"] += 1
            else:
                risultati["X (Pareggio)"] += 1
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
    else: # ft
        home_to_score_count = (df_to_score["home
