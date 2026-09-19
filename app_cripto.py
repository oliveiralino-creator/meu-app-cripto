"""
Crypto Market Intelligence — v2
--------------------------------
Mudanças estruturais em relação à v1:
  * UM único motor de score (compute_indicators) usado pelo Radar e pelo Backtesting.
    O que você calibra na aba 3 é exatamente o que a aba 1 recomenda.
  * RSI de Wilder vetorizado (a v1 devolvia um escalar e o backtest usava RSI constante).
  * Backtest com taxa/slippage, drawdown máximo, Sharpe, win rate, exposição e lista de trades.
  * Otimização walk-forward (treino/teste) para não sobreajustar os gatilhos ao passado.
  * Sentimento por ativo (notícias filtradas por moeda), com cache e saída JSON da IA.
  * Carteira virtual com fechamento de posição e P&L realizado/não realizado.
  * Watchlist persistente, retries com backoff e erros visíveis em vez de tela vazia.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import feedparser
import re
import os
import json
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

TZ = ZoneInfo("America/Sao_Paulo")
WATCHLIST_FILE = "watchlist.json"
CARTEIRA_FILE = "carteira_virtual.csv"
DEFAULT_WATCHLIST = ["SOL", "BTC", "ETH"]
PERIODS_PER_YEAR = {"daily": 365, "hourly": 365 * 24}

st.set_page_config(page_title="Crypto Market Intelligence", layout="wide", initial_sidebar_state="expanded")

# compatibilidade: Streamlit >= 1.49 usa width="stretch"; versões antigas, use_container_width
try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
    _W = {"width": "stretch"} if _v >= (1, 49) else {"use_container_width": True}
except Exception:
    _W = {"use_container_width": True}


# =====================================================================
# 1. CAMADA DE DADOS (com retry e erro explícito)
# =====================================================================
def _get_json(url, params=None, retries=3, timeout=15):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"accept": "application/json"})
            if r.status_code == 200:
                return r.json(), None
            if r.status_code == 429:
                last_err = "limite de requisições da API atingido (HTTP 429) — aguarde 1 minuto"
                time.sleep(2 ** i)
                continue
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = f"falha de rede: {e}"
        time.sleep(1)
    return None, last_err


@st.cache_data(ttl=300, show_spinner=False)
def get_market_overview():
    params = {
        "vs_currency": "usd", "order": "market_cap_desc", "per_page": 100, "page": 1,
        "sparkline": "true", "price_change_percentage": "1h,24h,7d,30d",
    }
    data, err = _get_json("https://api.coingecko.com/api/v3/coins/markets", params)
    return (data or []), err


@st.cache_data(ttl=3600, show_spinner=False)
def get_fear_and_greed():
    data, err = _get_json("https://api.alternative.me/fng/", {"limit": 1})
    if data:
        try:
            d = data["data"][0]
            return int(d["value"]), d["value_classification"], None
        except (KeyError, ValueError, IndexError) as e:
            err = f"formato inesperado: {e}"
    return 50, "Neutral", err


def _extract_ticker(data, ticker, single):
    """Isola o OHLCV de um ticker no retorno do yfinance (lida com MultiIndex)."""
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        if ticker in lvl0:
            d = data[ticker]
        elif single:
            d = data.droplevel(1, axis=1) if ticker in data.columns.get_level_values(1) else data.droplevel(0, axis=1)
        else:
            return None
    else:
        d = data
    d = d.dropna(subset=["Close"])
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in d.columns]
    return d[cols].copy()


@st.cache_data(ttl=900, show_spinner=False)
def get_daily_history(tickers: tuple, period: str = "1y"):
    """Histórico diário (yfinance) para uma tupla de tickers 'XXX-USD'."""
    out, missing = {}, []
    if not tickers:
        return out, missing, None
    try:
        data = yf.download(list(tickers), period=period, auto_adjust=True, progress=False,
                           group_by="ticker", threads=True)
    except Exception as e:
        return out, list(tickers), f"yfinance: {e}"
    if data is None or data.empty:
        return out, list(tickers), "yfinance não retornou dados"
    for t in tickers:
        try:
            d = _extract_ticker(data, t, single=(len(tickers) == 1))
            if d is not None and len(d) >= 30:
                out[t] = d
            else:
                missing.append(t)
        except Exception:
            missing.append(t)
    return out, missing, None


@st.cache_data(ttl=600, show_spinner=False)
def get_news(query: str = "", max_items: int = 15, days: int = 3):
    """query vazia → feed geral (Cointelegraph); com query → Google News pt-BR."""
    if not query:
        feed_url = "https://cointelegraph.com/rss"
    else:
        feed_url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(query)
                    + "&hl=pt-BR&gl=BR&ceid=BR:pt-419")
    news = []
    try:
        feed = feedparser.parse(feed_url)
        limite = datetime.now(timezone.utc) - timedelta(days=days)
        for e in feed.entries:
            dt_str = "Recente"
            if getattr(e, "published_parsed", None):
                dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                if dt < limite:
                    continue
                dt_str = dt.astimezone(TZ).strftime("%d/%m/%Y %H:%M")
            news.append({"title": e.title, "link": e.link, "published": dt_str})
            if len(news) >= max_items:
                break
    except Exception:
        pass
    return news


# =====================================================================
# 2. SENTIMENTO (léxico melhorado + IA com cache e saída JSON)
# =====================================================================
POS_WORDS = {"surge", "rally", "bull", "bullish", "jump", "gain", "gains", "adoption", "etf", "approve", "approves",
             "record", "upgrade", "partnership", "inflow", "inflows",
             "alta", "crescimento", "lucro", "dispara", "aprova", "aprovação", "recorde", "adoção", "valoriza",
             "sobe", "avança", "entrada", "parceria", "otimista"}
NEG_WORDS = {"plunge", "crash", "bear", "bearish", "drop", "fall", "falls", "hack", "hacked", "ban", "bans",
             "lawsuit", "sues", "exploit", "outflow", "outflows", "liquidation", "liquidations",
             "queda", "tombo", "roubo", "cai", "despenca", "hacker", "golpe", "processo", "proíbe", "saída",
             "liquidação", "liquidações", "pessimista", "desvaloriza", "recua"}
# frases resolvem ambiguidades como "SEC" (a v1 marcava toda menção à SEC como negativa)
POS_PHRASES = ["sec aprova", "sec approves", "aprova etf", "approves etf", "etf approval"]
NEG_PHRASES = ["sec processa", "sec sues", "sec rejeita", "sec rejects", "sec charges"]


def sentiment_lexical(news):
    """Média do sentimento por manchete (−3..+3) mapeada para 0..100. Sem notícias → 50."""
    if not news:
        return 50
    total = 0.0
    for n in news:
        t = n["title"].lower()
        s = 0
        s += 2 * sum(1 for p in POS_PHRASES if p in t)
        s -= 2 * sum(1 for p in NEG_PHRASES if p in t)
        words = set(re.findall(r"[a-záéíóúãõâêôç]+", t))
        s += len(words & POS_WORDS)
        s -= len(words & NEG_WORDS)
        total += max(-3, min(3, s))
    return int(round(50 + (total / len(news)) * (50 / 3)))


@st.cache_data(ttl=1800, show_spinner=False)
def sentiment_ai(titles: tuple, model_name: str, _api_key: str):
    """Chamada ao Gemini via google-genai. Cacheada pelo conjunto de manchetes (não pela chave)."""
    if not titles:
        return None, None, "sem manchetes"
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None, None, "biblioteca ausente: pip install google-genai"
    try:
        client = genai.Client(api_key=_api_key)
        prompt = (
            "Você é um analista quantitativo de criptomoedas. Avalie o conjunto de manchetes abaixo e responda "
            "com um score de sentimento de mercado de 0 (extremamente negativo) a 100 (extremamente positivo), "
            "onde 50 é neutro, e um resumo objetivo em português de no máximo 25 palavras.\n\nManchetes:\n"
            + "\n".join(f"- {t}" for t in titles)
        )
        schema = {"type": "OBJECT",
                  "properties": {"score": {"type": "INTEGER"}, "resumo": {"type": "STRING"}},
                  "required": ["score", "resumo"]}
        resp = client.models.generate_content(
            model=model_name, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema,
                                               temperature=0.2),
        )
        data = json.loads(resp.text)
        return int(max(0, min(100, int(data["score"])))), str(data["resumo"]).strip(), None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def get_sentiment(news, api_key, model_name):
    """Retorna (score, descrição). Usa IA se houver chave; cai para o léxico em qualquer falha."""
    lex = sentiment_lexical(news)
    if not api_key:
        return lex, "⚙️ IA desativada — sentimento léxico."
    score, resumo, err = sentiment_ai(tuple(n["title"] for n in news), model_name, api_key)
    if score is None:
        return lex, f"⚠️ IA indisponível ({err}) — usando léxico."
    return score, f"🧠 {resumo}"


# =====================================================================
# 3. MOTOR ÚNICO DE INDICADORES E SCORE
# =====================================================================
DEFAULT_PARAMS = dict(
    rsi_period=14, ema_fast=9, ema_slow=21, ma_long=50, mom_period=10, vol_window=20,
    w_rsi=0.30, w_trend=0.35, w_mom=0.20, w_vol=0.15,
)


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder (seed = média simples, depois suavização alpha=1/period). NaN durante o aquecimento."""
    close = pd.Series(close, dtype="float64")
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return pd.Series(out, index=close.index)
    delta = close.diff().values
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # seed clássico de Wilder: média simples das primeiras `period` variações; depois suavização alpha=1/period
    ag = gain[1:period + 1].mean()
    al = loss[1:period + 1].mean()
    out[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    k = 1.0 / period
    for i in range(period + 1, n):
        ag = ag + k * (gain[i] - ag)
        al = al + k * (loss[i] - al)
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return pd.Series(out, index=close.index)


def compute_indicators(close: pd.Series, volume: pd.Series | None = None, p: dict | None = None) -> pd.DataFrame:
    """
    Calcula indicadores e o Score técnico (0–100) para QUALQUER série de preços regular
    (diária no backtest e na watchlist; horária no sparkline do radar).
    Todos os indicadores olham apenas para trás → sem look-ahead.
    """
    p = {**DEFAULT_PARAMS, **(p or {})}
    df = pd.DataFrame({"Close": pd.Series(close, dtype="float64")})
    c = df["Close"]

    df["RSI"] = rsi_wilder(c, p["rsi_period"])
    df["EMA_F"] = c.ewm(span=p["ema_fast"], adjust=False).mean()
    df["EMA_S"] = c.ewm(span=p["ema_slow"], adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD"], df["Signal"] = macd, macd.ewm(span=9, adjust=False).mean()
    df["MA_L"] = c.rolling(p["ma_long"], min_periods=max(10, p["ma_long"] // 2)).mean()
    df["Mom"] = c.pct_change(p["mom_period"]) * 100
    df["Vol"] = c.pct_change().rolling(p["vol_window"], min_periods=10).std() * 100
    if volume is not None:
        v = pd.Series(volume, dtype="float64").reindex(df.index)
        df["VolZ"] = (v - v.rolling(20, min_periods=10).mean()) / v.rolling(20, min_periods=10).std()

    # --- componentes 0..100 ---
    # RSI contrarian: RSI<=30 → 100 (sobrevendido, oportunidade), RSI>=70 → 0
    df["S_RSI"] = (100 - (df["RSI"] - 30) * 2.5).clip(0, 100)
    # Tendência: EMA rápida > lenta (35), MACD > sinal (35), preço > média longa (30; 15 se ainda sem média)
    trend = np.where(df["EMA_F"] > df["EMA_S"], 35.0, 0.0)
    trend = trend + np.where(df["MACD"] > df["Signal"], 35.0, 0.0)
    trend = trend + np.where(df["MA_L"].isna(), 15.0, np.where(c > df["MA_L"], 30.0, 0.0))
    df["S_TREND"] = trend
    # Momentum: variação de -10%..+10% no período → 0..100
    df["S_MOM"] = ((df["Mom"] + 10) * 5).clip(0, 100)
    # Risco: menor volatilidade → maior score (desvio de 0% → 100; 10% → 0)
    df["S_VOL"] = (100 - df["Vol"] * 10).clip(0, 100)

    df["Score"] = (p["w_rsi"] * df["S_RSI"] + p["w_trend"] * df["S_TREND"]
                   + p["w_mom"] * df["S_MOM"] + p["w_vol"] * df["S_VOL"])
    return df


def composite_score(tech, sentiment, fg, w_tech=0.70, w_sent=0.15, w_fg=0.15):
    """Score final do radar: técnica + sentimento do ativo + Fear&Greed contrarian."""
    return float(np.clip(w_tech * tech + w_sent * sentiment + w_fg * (100 - fg), 0, 100))


def classify_action(score):
    if pd.isna(score):
        return "⚪ Sem dados"
    if score >= 80: return "🟢 Compra Forte"
    if score >= 60: return "🟢 Compra Média"
    if score >= 51: return "🟡 Compra Fraca"
    if score >= 45: return "⚪ Neutro"
    if score >= 41: return "🟠 Venda Fraca"
    if score >= 21: return "🔴 Venda Média"
    return "🔴 Venda Forte"


# =====================================================================
# 4. BACKTEST E OTIMIZAÇÃO WALK-FORWARD
# =====================================================================
def run_backtest(ind: pd.DataFrame, buy_thr: float, sell_thr: float, fee_pct: float = 0.1,
                 periods_per_year: int = 365):
    """
    Sinal gerado no fechamento de t, exposição vale a partir de t+1 (shift). Taxa cobrada a cada troca de posição.
    Retorna (DataFrame com equity, dict de métricas, DataFrame de trades).
    """
    df = ind.copy()
    sig = pd.Series(np.nan, index=df.index, dtype="float64")
    sig[df["Score"] >= buy_thr] = 1.0
    sig[df["Score"] <= sell_thr] = 0.0
    df["Pos"] = sig.ffill().fillna(0.0)
    df["Ret"] = df["Close"].pct_change().fillna(0.0)
    pos_prev = df["Pos"].shift(1).fillna(0.0)
    switch = (df["Pos"] != pos_prev).astype(float)
    switch.iloc[0] = 0.0
    df["Ret_Robo"] = df["Ret"] * pos_prev - switch * (fee_pct / 100.0)
    df["Eq_Hold"] = (1 + df["Ret"]).cumprod()
    df["Eq_Robo"] = (1 + df["Ret_Robo"]).cumprod()

    # lista de trades (round-trips)
    trades, entry = [], None
    pos, close, idx = df["Pos"].values, df["Close"].values, df.index
    for i in range(1, len(pos)):
        if pos[i - 1] == 0 and pos[i] == 1:
            entry = (idx[i], close[i])
        elif pos[i - 1] == 1 and pos[i] == 0 and entry:
            trades.append({"Entrada": entry[0], "Saída": idx[i], "Preço Entrada": entry[1], "Preço Saída": close[i],
                           "Retorno (%)": (close[i] / entry[1] - 1) * 100 - 2 * fee_pct, "Aberta": False})
            entry = None
    if entry:
        trades.append({"Entrada": entry[0], "Saída": idx[-1], "Preço Entrada": entry[1], "Preço Saída": close[-1],
                       "Retorno (%)": (close[-1] / entry[1] - 1) * 100 - fee_pct, "Aberta": True})
    trades_df = pd.DataFrame(trades)

    n = len(df)
    years = max(n / periods_per_year, 1e-9)
    ret_robo, ret_hold = df["Eq_Robo"].iloc[-1] - 1, df["Eq_Hold"].iloc[-1] - 1
    dd_robo = (df["Eq_Robo"] / df["Eq_Robo"].cummax() - 1).min()
    dd_hold = (df["Eq_Hold"] / df["Eq_Hold"].cummax() - 1).min()
    std = df["Ret_Robo"].std()
    sharpe = (df["Ret_Robo"].mean() / std * np.sqrt(periods_per_year)) if std and std > 0 else 0.0
    std_h = df["Ret"].std()
    sharpe_h = (df["Ret"].mean() / std_h * np.sqrt(periods_per_year)) if std_h and std_h > 0 else 0.0
    closed = trades_df[~trades_df["Aberta"]] if not trades_df.empty else trades_df
    metrics = {
        "ret_robo": ret_robo * 100, "ret_hold": ret_hold * 100,
        "cagr_robo": ((1 + ret_robo) ** (1 / years) - 1) * 100 if ret_robo > -1 else -100.0,
        "cagr_hold": ((1 + ret_hold) ** (1 / years) - 1) * 100 if ret_hold > -1 else -100.0,
        "dd_robo": dd_robo * 100, "dd_hold": dd_hold * 100,
        "sharpe_robo": sharpe, "sharpe_hold": sharpe_h,
        "n_trades": int(len(trades_df)),
        "win_rate": float((closed["Retorno (%)"] > 0).mean() * 100) if len(closed) else float("nan"),
        "exposure": df["Pos"].mean() * 100,
    }
    return df, metrics, trades_df


def optimize_walk_forward(ind: pd.DataFrame, buy_grid, sell_grid, fee_pct=0.1, train_frac=0.7,
                          periods_per_year=365):
    """Grid nos gatilhos: escolhe no treino (Sharpe) e reporta o desempenho fora da amostra (teste)."""
    cut = int(len(ind) * train_frac)
    train, test = ind.iloc[:cut], ind.iloc[cut:]
    rows = []
    for b in buy_grid:
        for s in sell_grid:
            if s >= b:
                continue
            _, m_tr, _ = run_backtest(train, b, s, fee_pct, periods_per_year)
            _, m_te, _ = run_backtest(test, b, s, fee_pct, periods_per_year)
            rows.append({"Compra ≥": b, "Venda ≤": s,
                         "Sharpe Treino": m_tr["sharpe_robo"], "Retorno Treino (%)": m_tr["ret_robo"],
                         "Sharpe Teste": m_te["sharpe_robo"], "Retorno Teste (%)": m_te["ret_robo"],
                         "DD Teste (%)": m_te["dd_robo"], "Trades Teste": m_te["n_trades"]})
    res = pd.DataFrame(rows).sort_values("Sharpe Treino", ascending=False).reset_index(drop=True)
    _, m_hold_te, _ = run_backtest(test, 101, -1, 0.0, periods_per_year)  # nunca compra → só buy&hold
    return res, m_hold_te, test.index[0]


# =====================================================================
# 5. PERSISTÊNCIA (watchlist e carteira)
# =====================================================================
def load_watchlist():
    try:
        with open(WATCHLIST_FILE) as f:
            wl = json.load(f)
            if isinstance(wl, list) and wl:
                return [str(x).upper() for x in wl]
    except Exception:
        pass
    return DEFAULT_WATCHLIST.copy()


def save_watchlist(wl):
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(wl, f)
        return True
    except Exception:
        return False


CARTEIRA_COLS = ["Data", "Ativo", "Preco_Compra", "Quantidade", "Taxa_pct", "Status", "Data_Venda", "Preco_Venda"]


def load_carteira():
    if not os.path.exists(CARTEIRA_FILE):
        return pd.DataFrame(columns=CARTEIRA_COLS)
    df = pd.read_csv(CARTEIRA_FILE)
    df = df.rename(columns={"Preço_Compra": "Preco_Compra"})  # compatibilidade com a v1
    for c, default in [("Taxa_pct", 0.0), ("Status", "Aberta"), ("Data_Venda", ""), ("Preco_Venda", np.nan)]:
        if c not in df.columns:
            df[c] = default
    return df[CARTEIRA_COLS]


def save_carteira(df):
    df[CARTEIRA_COLS].to_csv(CARTEIRA_FILE, index=False)


def fmt_price(v):
    if pd.isna(v):
        return "—"
    return f"${v:,.2f}" if v >= 1 else (f"${v:,.4f}" if v >= 0.01 else f"${v:,.6f}")


# =====================================================================
# 6. SIDEBAR
# =====================================================================
raw, err_cg = get_market_overview()
df_mkt = pd.DataFrame(raw) if raw else pd.DataFrame()
if not df_mkt.empty:
    df_mkt["Ativo"] = df_mkt["symbol"].str.upper()
    lista_ativos = df_mkt["Ativo"].tolist()
    nome_por_ativo = dict(zip(df_mkt["Ativo"], df_mkt["name"]))
    preco_por_ativo = dict(zip(df_mkt["Ativo"], df_mkt["current_price"]))
    var24h_por_ativo = dict(zip(df_mkt["Ativo"], df_mkt.get("price_change_percentage_24h_in_currency", np.nan)))
else:
    lista_ativos, nome_por_ativo, preco_por_ativo, var24h_por_ativo = [], {}, {}, {}

st.sidebar.title("⚙️ Configurações")
with st.sidebar.expander("🧠 IA (Gemini)", expanded=False):
    ia_key = st.text_input("API Key:", type="password")
    ia_model = st.text_input("Modelo:", value="gemini-2.5-flash")
    st.caption("Chamadas são cacheadas por 30 min por conjunto de manchetes.")

st.sidebar.divider()
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = load_watchlist()
opcoes_wl = sorted(set(lista_ativos) | set(st.session_state["watchlist"]))
watchlist = st.sidebar.multiselect("⭐ Watchlist (análise diária):", options=opcoes_wl,
                                   default=[a for a in st.session_state["watchlist"] if a in opcoes_wl],
                                   max_selections=10)
if st.sidebar.button("💾 Salvar watchlist"):
    st.session_state["watchlist"] = watchlist
    st.sidebar.success("Salva." if save_watchlist(watchlist) else "Não foi possível gravar o arquivo.")

st.sidebar.divider()
with st.sidebar.expander("🔧 Parâmetros do motor (radar + backtest)"):
    P = dict(DEFAULT_PARAMS)
    P["rsi_period"] = st.slider("Período RSI", 7, 28, P["rsi_period"])
    P["ema_fast"] = st.slider("EMA rápida", 5, 20, P["ema_fast"])
    P["ema_slow"] = st.slider("EMA lenta", 15, 60, P["ema_slow"])
    P["ma_long"] = st.slider("Média longa", 20, 200, P["ma_long"], step=10)
    P["mom_period"] = st.slider("Período momentum", 3, 30, P["mom_period"])
    st.markdown("**Pesos** (normalizados automaticamente)")
    w = [st.slider("RSI", 0.0, 1.0, P["w_rsi"], 0.05), st.slider("Tendência", 0.0, 1.0, P["w_trend"], 0.05),
         st.slider("Momentum", 0.0, 1.0, P["w_mom"], 0.05), st.slider("Risco (volatilidade)", 0.0, 1.0, P["w_vol"], 0.05)]
    tot = sum(w) or 1.0
    P["w_rsi"], P["w_trend"], P["w_mom"], P["w_vol"] = [x / tot for x in w]

if err_cg:
    st.sidebar.error(f"CoinGecko: {err_cg}")

# =====================================================================
# 7. ABAS
# =====================================================================
tab1, tab2, tab3 = st.tabs(["📊 Radar de Mercado", "💼 Simulador de Carteira", "⏪ Backtesting"])

# ---------------------------------------------------------------------
# ABA 1 — RADAR
# ---------------------------------------------------------------------
with tab1:
    fg_value, fg_class, err_fg = get_fear_and_greed()
    news_geral = get_news("")
    sent_geral, ia_status = get_sentiment(news_geral, ia_key, ia_model)

    c1, c2, c3 = st.columns(3)
    c1.metric("Fear & Greed (macro)", f"{fg_value}/100", fg_class)
    c2.metric("Sentimento geral (notícias)", f"{sent_geral}/100", "Positivo" if sent_geral > 50 else "Negativo")
    c3.info(ia_status)
    if err_fg:
        st.caption(f"⚠️ Fear & Greed indisponível ({err_fg}); usando 50.")
    st.divider()

    # ---------- Watchlist: análise DIÁRIA com o mesmo motor do backtest ----------
    if watchlist:
        st.subheader("⭐ Watchlist — análise diária (mesmo motor do backtesting)")
        tickers = tuple(f"{a}-USD" for a in watchlist)
        with st.spinner("Baixando histórico diário..."):
            hist_map, missing, err_yf = get_daily_history(tickers, "1y")
        if err_yf:
            st.warning(err_yf)
        if missing:
            st.caption("Sem histórico diário no yfinance: " + ", ".join(m.replace("-USD", "") for m in missing))

        rows, ind_map = [], {}
        for a in watchlist:
            t = f"{a}-USD"
            if t not in hist_map:
                continue
            h = hist_map[t]
            ind = compute_indicators(h["Close"], h.get("Volume"), P)
            ind_map[a] = ind
            last = ind.iloc[-1]
            news_a = get_news(f"{nome_por_ativo.get(a, a)} criptomoeda")
            sent_a = sentiment_lexical(news_a) if len(news_a) >= 3 else sent_geral
            final = composite_score(last["Score"], sent_a, fg_value)
            rows.append({
                "Ativo": a, "Preço (USD)": float(last["Close"]),
                "Var 24h (%)": float(var24h_por_ativo.get(a, np.nan)),
                "RSI": float(last["RSI"]), "Tendência": float(last["S_TREND"]), "Momentum": float(last["S_MOM"]),
                "Risco": float(last["S_VOL"]), "Sentimento": sent_a, "Score Técnico": float(last["Score"]),
                "Score Final": final, "Decisão": classify_action(final), "_news": len(news_a),
            })

        if rows:
            df_wl = pd.DataFrame(rows).sort_values("Score Final", ascending=False)
            st.dataframe(
                df_wl.drop(columns="_news").style
                .format({"Preço (USD)": fmt_price, "Var 24h (%)": "{:.2f}%", "RSI": "{:.0f}", "Tendência": "{:.0f}",
                         "Momentum": "{:.0f}", "Risco": "{:.0f}", "Sentimento": "{:.0f}",
                         "Score Técnico": "{:.0f}", "Score Final": "{:.0f}"})
                .background_gradient(subset=["Score Final"], cmap="RdYlGn", vmin=0, vmax=100),
                hide_index=True, **_W)

            col_g1, col_g2 = st.columns([3, 1])
            base100 = col_g2.toggle("Normalizar (base 100)", value=len(ind_map) > 1)
            dias = col_g2.select_slider("Janela", options=[30, 90, 180, 365], value=90)
            fig = go.Figure()
            for a, ind in ind_map.items():
                s = ind["Close"].iloc[-dias:]
                y = s / s.iloc[0] * 100 if base100 else s
                fig.add_trace(go.Scatter(x=s.index, y=y, mode="lines", name=a))
            fig.update_layout(template="plotly_white", height=380, margin=dict(t=30, b=10),
                              yaxis_title="Base 100" if base100 else "USD",
                              xaxis=dict(tickformat="%d/%m", hoverformat="%d/%m/%Y"))
            col_g1.plotly_chart(fig, **_W)

            with st.expander("🔎 Detalhe por ativo (score diário, RSI, tendência)"):
                a_det = st.selectbox("Ativo:", list(ind_map.keys()))
                ind = ind_map[a_det].iloc[-dias:]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=ind.index, y=ind["Score"], name="Score técnico", line=dict(color="green")))
                fig2.add_trace(go.Scatter(x=ind.index, y=ind["RSI"], name="RSI", line=dict(color="orange", dash="dot")))
                fig2.add_hline(y=50, line_dash="dash", line_color="gray")
                fig2.update_layout(template="plotly_white", height=300, margin=dict(t=20, b=10), yaxis_range=[0, 100])
                st.plotly_chart(fig2, **_W)
                nws = get_news(f"{nome_por_ativo.get(a_det, a_det)} criptomoeda")
                if nws:
                    st.markdown(f"**Notícias recentes — {a_det}** ({len(nws)})")
                    for n in nws[:8]:
                        st.markdown(f"- [{n['title']}]({n['link']}) — _{n['published']}_")
                else:
                    st.caption("Sem notícias específicas nos últimos 3 dias; sentimento geral aplicado.")
        st.divider()

    # ---------- Top 100: score rápido sobre o sparkline HORÁRIO de 7 dias ----------
    st.subheader("🌐 Top 100 — score rápido (base horária, últimos 7 dias)")
    st.caption("Mesmo motor, mas sobre 168 pontos horários do CoinGecko: RSI/EMAs em horas, momentum de 24h. "
               "Use a watchlist acima para a leitura diária, que é a validada pelo backtesting.")
    if df_mkt.empty:
        st.error("Sem dados do CoinGecko" + (f": {err_cg}" if err_cg else "."))
    else:
        P_H = {**P, "mom_period": 24}
        techs = []
        for _, r in df_mkt.iterrows():
            prices = (r.get("sparkline_in_7d") or {}).get("price") or []
            if len(prices) < 60:
                techs.append(np.nan)
                continue
            techs.append(float(compute_indicators(pd.Series(prices), None, P_H)["Score"].iloc[-1]))
        df_mkt["Score Técnico"] = techs
        df_mkt["Score Final"] = df_mkt["Score Técnico"].apply(
            lambda t: np.nan if pd.isna(t) else composite_score(t, sent_geral, fg_value))
        df_mkt["Decisão"] = df_mkt["Score Final"].apply(classify_action)
        df_mkt["Vol/Cap (%)"] = (df_mkt["total_volume"] / df_mkt["market_cap"].replace(0, np.nan) * 100).round(2)

        filtro = st.multiselect("Filtrar moedas:", options=lista_ativos, default=[])
        df_view = df_mkt[df_mkt["Ativo"].isin(filtro)] if filtro else df_mkt
        cols = {"Ativo": "Ativo", "current_price": "Preço (USD)",
                "price_change_percentage_1h_in_currency": "Var 1h (%)",
                "price_change_percentage_24h_in_currency": "Var 24h (%)",
                "price_change_percentage_7d_in_currency": "Var 7d (%)",
                "Vol/Cap (%)": "Vol/Cap (%)", "Score Técnico": "Score Técnico",
                "Score Final": "Score Final", "Decisão": "Decisão"}
        df_clean = df_view.rename(columns=cols)[list(cols.values())].sort_values("Score Final", ascending=False)
        st.dataframe(
            df_clean.style.format({"Preço (USD)": fmt_price, "Var 1h (%)": "{:.2f}%", "Var 24h (%)": "{:.2f}%",
                                   "Var 7d (%)": "{:.2f}%", "Vol/Cap (%)": "{:.1f}%",
                                   "Score Técnico": "{:.0f}", "Score Final": "{:.0f}"})
            .background_gradient(subset=["Score Final"], cmap="RdYlGn", vmin=0, vmax=100),
            hide_index=True, **_W)

    with st.expander("📰 Notícias gerais (Cointelegraph)"):
        for n in news_geral[:10]:
            st.markdown(f"- [{n['title']}]({n['link']}) — _{n['published']}_")

# ---------------------------------------------------------------------
# ABA 2 — CARTEIRA VIRTUAL
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Simulador de Posições (Forward-Testing)")
    st.caption("Arquivo local `carteira_virtual.csv`. Em hospedagem efêmera (Streamlit Cloud) o histórico se perde "
               "ao reiniciar — baixe o CSV pelo botão abaixo se quiser guardar.")
    df_cart = load_carteira()
    opcoes_compra = lista_ativos if lista_ativos else DEFAULT_WATCHLIST

    with st.form("form_compra"):
        c1, c2, c3, c4 = st.columns(4)
        ativo_sim = c1.selectbox("Ativo", options=opcoes_compra)
        preco_sug = float(preco_por_ativo.get(ativo_sim, 0.0) or 0.0)
        preco_compra = c2.number_input("Preço de compra (USD)", min_value=0.0, value=preco_sug, format="%.6f")
        quantidade = c3.number_input("Quantidade", min_value=0.0, value=1.0, format="%.4f")
        taxa = c4.number_input("Taxa (%)", min_value=0.0, value=0.1, step=0.05, format="%.2f")
        if st.form_submit_button("🛒 Registrar compra virtual") and quantidade > 0 and preco_compra > 0:
            nova = pd.DataFrame([{"Data": datetime.now(TZ).strftime("%Y-%m-%d %H:%M"), "Ativo": ativo_sim,
                                  "Preco_Compra": preco_compra, "Quantidade": quantidade, "Taxa_pct": taxa,
                                  "Status": "Aberta", "Data_Venda": "", "Preco_Venda": np.nan}])
            df_cart = pd.concat([df_cart, nova], ignore_index=True)
            save_carteira(df_cart)
            st.success(f"Posição de {quantidade} {ativo_sim} a {fmt_price(preco_compra)} registrada.")

    abertas = df_cart[df_cart["Status"] == "Aberta"].copy()
    if not abertas.empty:
        with st.form("form_venda"):
            st.markdown("**Fechar posição**")
            labels = {i: f"#{i} · {r.Ativo} · {r.Quantidade:g} @ {fmt_price(r.Preco_Compra)} ({r.Data})"
                      for i, r in abertas.iterrows()}
            v1, v2 = st.columns([3, 1])
            idx_fechar = v1.selectbox("Posição", options=list(labels.keys()), format_func=lambda i: labels[i])
            preco_sug_v = float(preco_por_ativo.get(abertas.loc[idx_fechar, "Ativo"], 0.0) or 0.0)
            preco_venda = v2.number_input("Preço de venda (USD)", min_value=0.0, value=preco_sug_v, format="%.6f")
            if st.form_submit_button("✅ Fechar posição") and preco_venda > 0:
                df_cart.loc[idx_fechar, ["Status", "Data_Venda", "Preco_Venda"]] = [
                    "Fechada", datetime.now(TZ).strftime("%Y-%m-%d %H:%M"), preco_venda]
                save_carteira(df_cart)
                st.success("Posição fechada.")
                st.rerun()

    st.divider()
    if not df_cart.empty:
        df_cart["Preco_Atual"] = df_cart["Ativo"].map(preco_por_ativo)
        df_cart["Investido"] = df_cart["Preco_Compra"] * df_cart["Quantidade"] * (1 + df_cart["Taxa_pct"] / 100)
        preco_ref = np.where(df_cart["Status"] == "Fechada", df_cart["Preco_Venda"], df_cart["Preco_Atual"])
        df_cart["Valor"] = preco_ref * df_cart["Quantidade"] * (1 - df_cart["Taxa_pct"] / 100)
        df_cart["P&L (USD)"] = df_cart["Valor"] - df_cart["Investido"]
        df_cart["Retorno (%)"] = df_cart["P&L (USD)"] / df_cart["Investido"] * 100

        nr = df_cart.loc[df_cart["Status"] == "Aberta", "P&L (USD)"].sum()
        rz = df_cart.loc[df_cart["Status"] == "Fechada", "P&L (USD)"].sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("P&L não realizado", f"${nr:,.2f}")
        m2.metric("P&L realizado", f"${rz:,.2f}")
        m3.metric("Total", f"${nr + rz:,.2f}")

        show = df_cart[["Data", "Ativo", "Status", "Quantidade", "Preco_Compra", "Preco_Atual", "Preco_Venda",
                        "Investido", "Valor", "P&L (USD)", "Retorno (%)"]]
        st.dataframe(show.style.format({"Preco_Compra": fmt_price, "Preco_Atual": fmt_price, "Preco_Venda": fmt_price,
                                        "Investido": "${:,.2f}", "Valor": "${:,.2f}", "P&L (USD)": "${:,.2f}",
                                        "Retorno (%)": "{:.2f}%", "Quantidade": "{:g}"})
                     .background_gradient(subset=["Retorno (%)"], cmap="RdYlGn", vmin=-10, vmax=10),
                     hide_index=True, **_W)
        b1, b2 = st.columns(2)
        b1.download_button("📥 Baixar carteira (CSV)", df_cart[CARTEIRA_COLS].to_csv(index=False).encode("utf-8"),
                           "carteira_virtual.csv", "text/csv")
        if b2.button("🗑️ Limpar carteira virtual"):
            os.remove(CARTEIRA_FILE)
            st.rerun()
    else:
        st.info("Nenhuma posição registrada.")

# ---------------------------------------------------------------------
# ABA 3 — BACKTESTING
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Backtesting do motor (o mesmo score do Radar)")
    st.caption("Indicadores olham só para trás; o sinal do fechamento de hoje vale a partir de amanhã. "
               "Taxa cobrada em cada entrada e saída. Otimize com walk-forward para ver o resultado fora da amostra.")

    cA, cB, cC, cD = st.columns(4)
    opcoes_bt = sorted(set(watchlist) | {"BTC", "ETH", "SOL"})
    ativo_bt = cA.selectbox("Ativo:", opcoes_bt + ["Outro..."])
    if ativo_bt == "Outro...":
        ativo_bt = cA.text_input("Símbolo (ex.: AVAX):", value="AVAX").upper().strip()
    periodo_bt = cB.selectbox("Período:", ["1y", "6mo", "2y", "5y", "max"])
    fee_bt = cC.number_input("Taxa por operação (%)", 0.0, 2.0, 0.10, 0.05, format="%.2f")
    train_frac = cD.slider("Fatia de treino (walk-forward)", 0.5, 0.9, 0.7, 0.05)

    s1, s2 = st.columns(2)
    gatilho_compra = s1.slider("Gatilho de compra (Score ≥):", 50, 90, 65, 5)
    gatilho_venda = s2.slider("Gatilho de venda (Score ≤):", 20, 60, 45, 5)
    if gatilho_venda >= gatilho_compra:
        st.error("O gatilho de venda precisa ser menor que o de compra.")

    b1, b2 = st.columns(2)
    rodar = b1.button("▶️ Rodar simulação", disabled=gatilho_venda >= gatilho_compra)
    otimizar = b2.button("🧪 Otimizar gatilhos (walk-forward)")

    if rodar or otimizar:
        ticker = f"{ativo_bt}-USD"
        with st.spinner(f"Baixando {ticker}..."):
            hist_map, missing, err_yf = get_daily_history((ticker,), periodo_bt)
        if ticker not in hist_map:
            st.error(f"Sem dados para {ticker}" + (f": {err_yf}" if err_yf else " (ticker inexistente no yfinance?)"))
        else:
            h = hist_map[ticker]
            ind = compute_indicators(h["Close"], h.get("Volume"), P).dropna(subset=["Score"])

            if rodar:
                res, m, trades = run_backtest(ind, gatilho_compra, gatilho_venda, fee_bt)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res.index, y=(res["Eq_Hold"] - 1) * 100, name="Buy & Hold (%)",
                                         line=dict(color="gray")))
                fig.add_trace(go.Scatter(x=res.index, y=(res["Eq_Robo"] - 1) * 100, name="Robô (%)",
                                         line=dict(color="green", width=2)))
                comprado = res[res["Pos"] == 1]
                fig.add_trace(go.Scatter(x=comprado.index, y=(comprado["Eq_Robo"] - 1) * 100, mode="markers",
                                         marker=dict(size=3, color="green"), name="Em posição", showlegend=False))
                fig.update_layout(title=f"{ativo_bt}: compra ≥ {gatilho_compra}, venda ≤ {gatilho_venda}, taxa {fee_bt}%",
                                  yaxis_title="Retorno acumulado (%)", template="plotly_white", height=420)
                st.plotly_chart(fig, **_W)

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Retorno robô", f"{m['ret_robo']:.1f}%", f"{m['ret_robo'] - m['ret_hold']:+.1f} pp vs hold")
                k2.metric("Drawdown máx.", f"{m['dd_robo']:.1f}%", f"hold: {m['dd_hold']:.1f}%", delta_color="off")
                k3.metric("Sharpe (anual.)", f"{m['sharpe_robo']:.2f}", f"hold: {m['sharpe_hold']:.2f}", delta_color="off")
                wr = "—" if np.isnan(m["win_rate"]) else f"{m['win_rate']:.0f}%"
                k4.metric("Trades / win rate", f"{m['n_trades']} / {wr}", f"exposição {m['exposure']:.0f}%",
                          delta_color="off")
                k5, k6 = st.columns(2)
                k5.metric("CAGR robô", f"{m['cagr_robo']:.1f}%")
                k6.metric("CAGR hold", f"{m['cagr_hold']:.1f}%")

                if not trades.empty:
                    with st.expander(f"📋 Trades ({len(trades)})"):
                        t = trades.copy()
                        t["Entrada"] = pd.to_datetime(t["Entrada"]).dt.strftime("%d/%m/%Y")
                        t["Saída"] = pd.to_datetime(t["Saída"]).dt.strftime("%d/%m/%Y")
                        st.dataframe(t.style.format({"Preço Entrada": fmt_price, "Preço Saída": fmt_price,
                                                     "Retorno (%)": "{:.2f}%"})
                                     .background_gradient(subset=["Retorno (%)"], cmap="RdYlGn", vmin=-15, vmax=15),
                                     hide_index=True, **_W)
                export = res[["Close", "RSI", "EMA_F", "EMA_S", "MACD", "Signal", "MA_L", "Mom", "Vol",
                              "S_RSI", "S_TREND", "S_MOM", "S_VOL", "Score", "Pos", "Ret_Robo", "Eq_Robo",
                              "Eq_Hold"]].round(4)
                st.download_button("📥 Baixar série auditável (CSV)", export.to_csv().encode("utf-8"),
                                   f"backtest_{ativo_bt}_{periodo_bt}.csv", "text/csv")

            if otimizar:
                with st.spinner("Rodando grade de gatilhos..."):
                    grid, m_hold, inicio_teste = optimize_walk_forward(
                        ind, buy_grid=range(50, 91, 5), sell_grid=range(20, 61, 5),
                        fee_pct=fee_bt, train_frac=train_frac)
                st.markdown(f"**Treino:** até {(inicio_teste - pd.Timedelta(days=1)).strftime('%d/%m/%Y')} · "
                            f"**Teste (fora da amostra):** a partir de {inicio_teste.strftime('%d/%m/%Y')} · "
                            f"Buy & Hold no teste: **{m_hold['ret_hold']:.1f}%** (Sharpe {m_hold['sharpe_hold']:.2f})")
                st.caption("Ordenado pelo Sharpe no treino. O que importa é a coluna de TESTE: se os melhores do "
                           "treino não se sustentam no teste, os gatilhos estão sobreajustados.")
                st.dataframe(
                    grid.head(15).style
                    .format({"Sharpe Treino": "{:.2f}", "Retorno Treino (%)": "{:.1f}%", "Sharpe Teste": "{:.2f}",
                             "Retorno Teste (%)": "{:.1f}%", "DD Teste (%)": "{:.1f}%"})
                    .background_gradient(subset=["Sharpe Teste"], cmap="RdYlGn", vmin=-2, vmax=3),
                    hide_index=True, **_W)
                fig_hm = px.density_heatmap(grid, x="Compra ≥", y="Venda ≤", z="Sharpe Teste", histfunc="avg",
                                            color_continuous_scale="RdYlGn", template="plotly_white",
                                            title="Sharpe fora da amostra por combinação de gatilhos")
                fig_hm.update_layout(height=380)
                st.plotly_chart(fig_hm, **_W)
