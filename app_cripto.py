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
v3 (após diagnóstico em SOL 1y):
  * Momentum extremo penalizado no score (validado: SOL 5y +361% → +778%, permutação p=0,007).
  * Flag "Esticado" (aviso). O bloqueio de compra "não comprar esticado" existe mas fica DESLIGADO:
    em SOL 5y piorou in-sample e fora da amostra — bloqueia os rompimentos que pagam a estratégia.
  * Filtro de regime BTC (MA200): sem posição comprada quando o BTC está abaixo da média longa.
    Em SOL 5y: menos drawdown (−54% → −38%) e 2022 positivo, mas custa retorno fora da amostra
    (OOS +226% → +104%; SOL sai do fundo antes do BTC). MA150/100 pioram ainda mais o OOS.
    É um seguro, não um amplificador — DESLIGADO por padrão.
  * Walk-forward rolante (equity 100% fora da amostra), diagnóstico do sinal por componente e
    teste de permutação (o resultado é distinguível de sorte?).
  * Padrões = configuração validada; a sidebar salva/carrega parâmetros em parametros.json.
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
    # pesos validados em SOL 5y (permutação p=0,007; walk-forward rolante OOS +226% vs hold +130%)
    w_rsi=0.10, w_trend=0.35, w_mom=0.20, w_vol=0.35,
    mom_extreme=15.0,   # acima deste % no período, o momentum passa a ser penalizado (esticado)
    rsi_extreme=75.0,   # RSI acima disto marca o ativo como esticado
)
DEFAULT_SETTINGS = dict(use_regime=False, regime_ma=200, skip_stretched=False, buy_thr=65, sell_thr=45, fee=0.10)
PARAMS_FILE = "parametros.json"


def load_params():
    try:
        with open(PARAMS_FILE) as f:
            saved = json.load(f)
        p = {**DEFAULT_PARAMS, **{k: v for k, v in saved.get("params", {}).items() if k in DEFAULT_PARAMS}}
        s = {**DEFAULT_SETTINGS, **{k: v for k, v in saved.get("settings", {}).items() if k in DEFAULT_SETTINGS}}
        return p, s, True
    except Exception:
        return dict(DEFAULT_PARAMS), dict(DEFAULT_SETTINGS), False


def save_params(p, s):
    try:
        with open(PARAMS_FILE, "w") as f:
            json.dump({"params": p, "settings": s}, f, indent=2)
        return True
    except Exception:
        return False


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
    # Momentum: variação de -10%..+10% no período → 0..100; acima de `mom_extreme` o score DECAI
    # (o diagnóstico em SOL mostrou retorno futuro negativo quando o momentum está esticado)
    s_mom = ((df["Mom"] + 10) * 5).clip(0, 100)
    decay = (100 - (df["Mom"] - p["mom_extreme"]) * 5).clip(0, 100)
    df["S_MOM"] = np.where(df["Mom"] > p["mom_extreme"], np.minimum(s_mom, decay), s_mom)
    # Esticado: momentum ou RSI extremos → não perseguir a compra
    df["Esticado"] = (df["Mom"] > p["mom_extreme"]) | (df["RSI"] > p["rsi_extreme"])
    # Risco: menor volatilidade → maior score (desvio de 0% → 100; 10% → 0)
    df["S_VOL"] = (100 - df["Vol"] * 10).clip(0, 100)

    df["Score"] = (p["w_rsi"] * df["S_RSI"] + p["w_trend"] * df["S_TREND"]
                   + p["w_mom"] * df["S_MOM"] + p["w_vol"] * df["S_VOL"])
    return df


def composite_score(tech, sentiment, fg, w_tech=0.70, w_sent=0.15, w_fg=0.15):
    """Score final do radar: técnica + sentimento do ativo + Fear&Greed contrarian."""
    return float(np.clip(w_tech * tech + w_sent * sentiment + w_fg * (100 - fg), 0, 100))


def btc_regime(btc_close: pd.Series, ma: int = 200) -> pd.Series:
    """True quando BTC fecha acima da sua média longa (regime altista). Só olha para trás."""
    m = btc_close.rolling(ma, min_periods=ma).mean()
    return (btc_close > m).where(m.notna())


def classify_action(score, esticado=False, regime_ok=True):
    if pd.isna(score):
        return "⚪ Sem dados"
    if score >= 60 and not regime_ok:
        return "⏸️ Aguardar (BTC abaixo da MA200)"
    if score >= 60 and esticado:
        return "🟠 Esticado — não perseguir"
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
                 periods_per_year: int = 365, regime: pd.Series | None = None, skip_stretched: bool = False):
    """
    Sinal gerado no fechamento de t, exposição vale a partir de t+1 (shift). Taxa cobrada a cada troca de posição.
    regime: série booleana (True = pode estar comprado); fora do regime a posição é zerada.
    skip_stretched: não abre compra quando o ativo está esticado (mas mantém posição já aberta).
    Retorna (DataFrame com equity, dict de métricas, DataFrame de trades).
    """
    df = ind.copy()
    buy = df["Score"] >= buy_thr
    if skip_stretched and "Esticado" in df:
        buy = buy & ~df["Esticado"].fillna(False).astype(bool)
    sig = pd.Series(np.nan, index=df.index, dtype="float64")
    sig[buy] = 1.0
    sig[df["Score"] <= sell_thr] = 0.0
    df["Pos"] = sig.ffill().fillna(0.0)
    if regime is not None:
        r = regime.reindex(df.index).ffill().fillna(True).astype(bool)   # sem histórico de BTC → não filtra
        df["Regime"] = r
        df["Pos"] = df["Pos"].where(r, 0.0)
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
                          periods_per_year=365, **bt_kw):
    """Grid nos gatilhos: escolhe no treino (Sharpe) e reporta o desempenho fora da amostra (teste)."""
    cut = int(len(ind) * train_frac)
    train, test = ind.iloc[:cut], ind.iloc[cut:]
    rows = []
    for b in buy_grid:
        for s in sell_grid:
            if s >= b:
                continue
            _, m_tr, _ = run_backtest(train, b, s, fee_pct, periods_per_year, **bt_kw)
            _, m_te, _ = run_backtest(test, b, s, fee_pct, periods_per_year, **bt_kw)
            rows.append({"Compra ≥": b, "Venda ≤": s,
                         "Sharpe Treino": m_tr["sharpe_robo"], "Retorno Treino (%)": m_tr["ret_robo"],
                         "Sharpe Teste": m_te["sharpe_robo"], "Retorno Teste (%)": m_te["ret_robo"],
                         "DD Teste (%)": m_te["dd_robo"], "Trades Teste": m_te["n_trades"]})
    res = pd.DataFrame(rows).sort_values("Sharpe Treino", ascending=False).reset_index(drop=True)
    _, m_hold_te, _ = run_backtest(test, 101, -1, 0.0, periods_per_year)  # nunca compra → só buy&hold
    return res, m_hold_te, test.index[0]


def _sharpe(r: pd.Series, ppy: int = 365) -> float:
    s = r.std()
    return float(r.mean() / s * np.sqrt(ppy)) if s and s > 0 else 0.0


def rolling_walk_forward(ind: pd.DataFrame, buy_grid, sell_grid, fee_pct=0.1, train_len=252, test_len=63,
                         periods_per_year=365, **bt_kw):
    """
    Walk-forward rolante: em cada janela escolhe (compra, venda) pelo Sharpe no treino e aplica no teste seguinte.
    Concatena os retornos de teste → curva de equity 100% fora da amostra.
    Retorna (tabela por janela, série de retornos OOS do robô, série de retornos OOS do hold).
    """
    combos = [(b, s) for b in buy_grid for s in sell_grid if s < b]
    rows, oos_robo, oos_hold = [], [], []
    start = 0
    while start + train_len + test_len <= len(ind):
        train = ind.iloc[start:start + train_len]
        test = ind.iloc[start + train_len:start + train_len + test_len]
        best, best_sh = None, -np.inf
        for b, s in combos:
            _, m, _ = run_backtest(train, b, s, fee_pct, periods_per_year, **bt_kw)
            if m["sharpe_robo"] > best_sh:
                best, best_sh = (b, s), m["sharpe_robo"]
        res, m_te, _ = run_backtest(test, best[0], best[1], fee_pct, periods_per_year, **bt_kw)
        oos_robo.append(res["Ret_Robo"]); oos_hold.append(res["Ret"])
        rows.append({"Janela": len(rows) + 1, "Treino até": train.index[-1], "Teste de": test.index[0],
                     "Teste até": test.index[-1], "Compra ≥": best[0], "Venda ≤": best[1],
                     "Sharpe Treino": best_sh, "Retorno Teste (%)": m_te["ret_robo"],
                     "Hold Teste (%)": m_te["ret_hold"], "Trades": m_te["n_trades"]})
        start += test_len
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
    return pd.DataFrame(rows), pd.concat(oos_robo), pd.concat(oos_hold)


def signal_diagnostics(ind: pd.DataFrame, horizon: int = 10):
    """Correlação (Spearman) de cada componente com o retorno futuro e retorno médio por faixa de score."""
    df = ind.copy()
    df["fwd"] = df["Close"].shift(-horizon) / df["Close"] - 1
    d = df.dropna(subset=["fwd", "Score"])
    comps = {"Score": "Score total", "S_RSI": "RSI (contrarian)", "S_TREND": "Tendência",
             "S_MOM": "Momentum", "S_VOL": "Risco (baixa vol)"}
    fwd_rank = d["fwd"].rank()   # Spearman = Pearson dos ranks (sem depender do scipy)
    corr = pd.DataFrame([{"Componente": lbl, "Spearman": d[c].rank().corr(fwd_rank),
                          "Pearson": d[c].corr(d["fwd"])} for c, lbl in comps.items() if c in d])
    bins = [0, 35, 45, 55, 65, 75, 100]
    labels = ["≤35", "35–45", "45–55", "55–65", "65–75", ">75"]
    by = d.groupby(pd.cut(d["Score"], bins, labels=labels), observed=False)["fwd"].agg(
        Retorno_medio="mean", Acerto=lambda s: (s > 0).mean(), N="count").reset_index()
    by.columns = ["Faixa de Score", f"Retorno médio {horizon}d (%)", "% dias positivos", "N"]
    by[f"Retorno médio {horizon}d (%)"] *= 100
    by["% dias positivos"] *= 100
    return corr, by, len(d)


def permutation_test(close: pd.Series, p: dict, buy_thr, sell_thr, fee_pct, n_iter=200, seed=0, **bt_kw):
    """
    Embaralha os retornos diários (destrói a estrutura temporal, preserva a distribuição), recalcula
    indicadores e roda o robô. Retorna (Sharpe real, array de Sharpes embaralhados, p-valor).
    """
    rng = np.random.default_rng(seed)
    ind = compute_indicators(close, None, p).dropna(subset=["Score"])
    _, m_real, _ = run_backtest(ind, buy_thr, sell_thr, fee_pct, **bt_kw)
    rets = close.pct_change().dropna().values
    sharpes = []
    for _ in range(n_iter):
        shuffled = pd.Series(close.iloc[0] * np.cumprod(1 + rng.permutation(rets)), index=close.index[1:])
        ind_s = compute_indicators(shuffled, None, p).dropna(subset=["Score"])
        _, m, _ = run_backtest(ind_s, buy_thr, sell_thr, fee_pct, **bt_kw)
        sharpes.append(m["sharpe_robo"])
    sharpes = np.array(sharpes)
    pval = float((sharpes >= m_real["sharpe_robo"]).mean())
    return m_real["sharpe_robo"], sharpes, pval


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
P0, S0, params_from_file = load_params()
with st.sidebar.expander("🔧 Parâmetros do motor (radar + backtest)"):
    P = dict(P0)
    P["rsi_period"] = st.slider("Período RSI", 7, 28, int(P["rsi_period"]))
    P["ema_fast"] = st.slider("EMA rápida", 5, 20, int(P["ema_fast"]))
    P["ema_slow"] = st.slider("EMA lenta", 15, 60, int(P["ema_slow"]))
    P["ma_long"] = st.slider("Média longa", 20, 200, int(P["ma_long"]), step=10)
    P["mom_period"] = st.slider("Período momentum", 3, 30, int(P["mom_period"]))
    st.markdown("**Pesos** (normalizados automaticamente)")
    w = [st.slider("RSI", 0.0, 1.0, float(P["w_rsi"]), 0.05), st.slider("Tendência", 0.0, 1.0, float(P["w_trend"]), 0.05),
         st.slider("Momentum", 0.0, 1.0, float(P["w_mom"]), 0.05),
         st.slider("Risco (volatilidade)", 0.0, 1.0, float(P["w_vol"]), 0.05)]
    tot = sum(w) or 1.0
    P["w_rsi"], P["w_trend"], P["w_mom"], P["w_vol"] = [round(x / tot, 4) for x in w]
    st.markdown("**Esticado**")
    P["mom_extreme"] = st.slider("Momentum extremo (%)", 5.0, 40.0, float(P["mom_extreme"]), 1.0,
                                 help="Acima disto o score de momentum decai e o ativo é marcado como esticado.")
    P["rsi_extreme"] = st.slider("RSI extremo", 65.0, 90.0, float(P["rsi_extreme"]), 1.0)

with st.sidebar.expander("🛡️ Filtros de risco", expanded=True):
    use_regime = st.toggle("Filtro de regime BTC", value=bool(S0["use_regime"]),
                           help="Só permite posição comprada quando o BTC está acima da sua média longa. "
                                "DESLIGADO por padrão: em SOL 5y reduz o drawdown, mas custa ~metade do retorno fora "
                                "da amostra (SOL sai do fundo antes do BTC). Ligue se preferir menos risco.")
    regime_ma = st.slider("Média do regime (dias)", 100, 200, int(S0["regime_ma"]), 10,
                          help="Médias mais curtas pioraram fora da amostra em SOL 5y; 200 é a melhor variante.")
    skip_stretched = st.toggle("Não comprar esticado", value=bool(S0["skip_stretched"]),
                               help="Bloqueia novas compras quando momentum/RSI estão extremos. DESLIGADO por padrão: em SOL 5y "
                                    "piorou o resultado in-sample e fora da amostra (bloqueia exatamente os rompimentos). "
                                    "O score já penaliza momentum extremo de forma suave.")

st.sidebar.divider()
st.sidebar.caption(("📂 Parâmetros carregados de `parametros.json`." if params_from_file
                    else "Usando padrões validados (SOL 5y).") +
                   " Salve para que sobrevivam ao recarregar o app.")
if st.sidebar.button("💾 Salvar parâmetros e filtros"):
    S_now = {"use_regime": use_regime, "regime_ma": regime_ma, "skip_stretched": skip_stretched,
             "buy_thr": st.session_state.get("bt_buy", S0["buy_thr"]),
             "sell_thr": st.session_state.get("bt_sell", S0["sell_thr"]),
             "fee": st.session_state.get("bt_fee", S0["fee"])}
    st.sidebar.success("Salvo." if save_params(P, S_now) else "Não foi possível gravar o arquivo.")
if st.sidebar.button("↩️ Restaurar padrões validados"):
    try:
        os.remove(PARAMS_FILE)
    except OSError:
        pass
    st.rerun()

if err_cg:
    st.sidebar.error(f"CoinGecko: {err_cg}")

# regime do BTC (histórico longo para a MA200 existir desde o início do backtest)
REGIME = None
regime_now, regime_err = True, None
if use_regime:
    _btc, _miss, regime_err = get_daily_history(("BTC-USD",), "5y")
    if "BTC-USD" in _btc:
        REGIME = btc_regime(_btc["BTC-USD"]["Close"], regime_ma)
        _last = REGIME.dropna()
        regime_now = bool(_last.iloc[-1]) if len(_last) else True
    else:
        regime_err = regime_err or "sem histórico do BTC — filtro de regime desativado nesta sessão"
BT_KW = {"regime": REGIME, "skip_stretched": skip_stretched}

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fear & Greed (macro)", f"{fg_value}/100", fg_class)
    c2.metric("Sentimento geral (notícias)", f"{sent_geral}/100", "Positivo" if sent_geral > 50 else "Negativo")
    if use_regime and REGIME is not None:
        c3.metric(f"Regime BTC (MA{regime_ma})", "🟢 Altista" if regime_now else "🔴 Baixista",
                  "compras liberadas" if regime_now else "compras bloqueadas", delta_color="off")
    else:
        c3.metric("Regime BTC", "desligado", regime_err or "", delta_color="off")
    c4.info(ia_status)
    if err_fg:
        st.caption(f"⚠️ Fear & Greed indisponível ({err_fg}); usando 50.")
    if use_regime and regime_err:
        st.caption(f"⚠️ {regime_err}")
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
                "Score Final": final, "Esticado": "⚠️" if bool(last["Esticado"]) else "",
                "Decisão": classify_action(final, bool(last["Esticado"]) and skip_stretched,
                                           regime_now or not use_regime),
                "_news": len(news_a),
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
        techs, stretched = [], []
        for _, r in df_mkt.iterrows():
            prices = (r.get("sparkline_in_7d") or {}).get("price") or []
            if len(prices) < 60:
                techs.append(np.nan); stretched.append(False)
                continue
            last_h = compute_indicators(pd.Series(prices), None, P_H).iloc[-1]
            techs.append(float(last_h["Score"])); stretched.append(bool(last_h["Esticado"]))
        df_mkt["Score Técnico"] = techs
        df_mkt["Score Final"] = df_mkt["Score Técnico"].apply(
            lambda t: np.nan if pd.isna(t) else composite_score(t, sent_geral, fg_value))
        df_mkt["Decisão"] = [classify_action(s, e and skip_stretched, regime_now or not use_regime)
                             for s, e in zip(df_mkt["Score Final"], stretched)]
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
    fee_bt = cC.number_input("Taxa por operação (%)", 0.0, 2.0, float(S0["fee"]), 0.05, format="%.2f", key="bt_fee")
    train_frac = cD.slider("Fatia de treino (walk-forward simples)", 0.5, 0.9, 0.7, 0.05)
    st.session_state["diag_hz"] = cD.select_slider("Horizonte do diagnóstico (dias)", options=[5, 10, 20], value=10)

    s1, s2 = st.columns(2)
    gatilho_compra = s1.slider("Gatilho de compra (Score ≥):", 50, 90, int(S0["buy_thr"]), 5, key="bt_buy")
    gatilho_venda = s2.slider("Gatilho de venda (Score ≤):", 20, 60, int(S0["sell_thr"]), 5, key="bt_sell")
    if gatilho_venda >= gatilho_compra:
        st.error("O gatilho de venda precisa ser menor que o de compra.")

    filtros_txt = " · ".join(f for f, on in [("regime BTC", use_regime and REGIME is not None),
                                              ("não comprar esticado", skip_stretched)] if on) or "nenhum"
    st.caption(f"Filtros ativos (sidebar): **{filtros_txt}**")

    b1, b2, b3, b4, b5 = st.columns(5)
    rodar = b1.button("▶️ Rodar simulação", disabled=gatilho_venda >= gatilho_compra)
    otimizar = b2.button("🧪 Walk-forward simples")
    rolante = b3.button("🔁 Walk-forward rolante")
    diagnosticar = b4.button("🩺 Diagnóstico do sinal")
    permutar = b5.button("🎲 Teste de permutação")

    if rodar or otimizar or rolante or diagnosticar or permutar:
        ticker = f"{ativo_bt}-USD"
        with st.spinner(f"Baixando {ticker}..."):
            hist_map, missing, err_yf = get_daily_history((ticker,), periodo_bt)
        if ticker not in hist_map:
            st.error(f"Sem dados para {ticker}" + (f": {err_yf}" if err_yf else " (ticker inexistente no yfinance?)"))
        else:
            h = hist_map[ticker]
            ind = compute_indicators(h["Close"], h.get("Volume"), P).dropna(subset=["Score"])

            if rodar:
                res, m, trades = run_backtest(ind, gatilho_compra, gatilho_venda, fee_bt, **BT_KW)
                # comparação: mesmo robô sem filtros
                _, m_raw, _ = run_backtest(ind, gatilho_compra, gatilho_venda, fee_bt)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res.index, y=(res["Eq_Hold"] - 1) * 100, name="Buy & Hold (%)",
                                         line=dict(color="gray")))
                fig.add_trace(go.Scatter(x=res.index, y=(res["Eq_Robo"] - 1) * 100, name="Robô (%)",
                                         line=dict(color="green", width=2)))
                if "Regime" in res:
                    off = res[~res["Regime"]]
                    if not off.empty:
                        fig.add_trace(go.Scatter(x=off.index, y=(off["Eq_Hold"] - 1) * 100, mode="markers",
                                                 marker=dict(size=3, color="red"), name="BTC em regime baixista"))
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
                k5, k6, k7 = st.columns(3)
                k5.metric("CAGR robô", f"{m['cagr_robo']:.1f}%")
                k6.metric("CAGR hold", f"{m['cagr_hold']:.1f}%")
                if BT_KW["regime"] is not None or BT_KW["skip_stretched"]:
                    k7.metric("Sem filtros (Sharpe / retorno)", f"{m_raw['sharpe_robo']:.2f} / {m_raw['ret_robo']:.1f}%",
                              f"filtros: {m['sharpe_robo'] - m_raw['sharpe_robo']:+.2f} de Sharpe")
                if m["n_trades"] < 8:
                    st.warning(f"Apenas {m['n_trades']} trade(s): amostra pequena demais para concluir. "
                               "Use um período maior, o walk-forward rolante e o teste de permutação.")

                if not trades.empty:
                    with st.expander(f"📋 Trades ({len(trades)})"):
                        t = trades.copy()
                        t["Entrada"] = pd.to_datetime(t["Entrada"]).dt.strftime("%d/%m/%Y")
                        t["Saída"] = pd.to_datetime(t["Saída"]).dt.strftime("%d/%m/%Y")
                        st.dataframe(t.style.format({"Preço Entrada": fmt_price, "Preço Saída": fmt_price,
                                                     "Retorno (%)": "{:.2f}%"})
                                     .background_gradient(subset=["Retorno (%)"], cmap="RdYlGn", vmin=-15, vmax=15),
                                     hide_index=True, **_W)
                exp_cols = ["Close", "RSI", "EMA_F", "EMA_S", "MACD", "Signal", "MA_L", "Mom", "Vol",
                            "S_RSI", "S_TREND", "S_MOM", "S_VOL", "Score", "Esticado", "Pos", "Ret_Robo",
                            "Eq_Robo", "Eq_Hold"] + (["Regime"] if "Regime" in res else [])
                export = res[exp_cols].round(4)
                st.download_button("📥 Baixar série auditável (CSV)", export.to_csv().encode("utf-8"),
                                   f"backtest_{ativo_bt}_{periodo_bt}.csv", "text/csv")

            if otimizar:
                with st.spinner("Rodando grade de gatilhos..."):
                    grid, m_hold, inicio_teste = optimize_walk_forward(
                        ind, buy_grid=range(50, 91, 5), sell_grid=range(20, 61, 5),
                        fee_pct=fee_bt, train_frac=train_frac, **BT_KW)
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

            if rolante:
                st.markdown("### 🔁 Walk-forward rolante (100% fora da amostra)")
                st.caption("Treina numa janela, escolhe os gatilhos, aplica no trimestre seguinte e avança. "
                           "A curva abaixo só usa decisões tomadas sem conhecer o futuro. "
                           "Gatilhos que mudam muito entre janelas = sinal instável.")
                train_len, test_len = 252, 63
                if len(ind) < train_len + 2 * test_len:
                    st.warning(f"Período curto ({len(ind)} dias) para janelas de {train_len}+{test_len}. "
                               "Use 2y ou mais; abaixo, janelas reduzidas.")
                    train_len, test_len = max(90, len(ind) // 3), max(30, len(ind) // 8)
                with st.spinner("Rodando janelas..."):
                    tab_wf, oos_r, oos_h = rolling_walk_forward(
                        ind, range(50, 91, 5), range(20, 61, 5), fee_bt, train_len, test_len, **BT_KW)
                if tab_wf.empty:
                    st.error("Dados insuficientes para ao menos uma janela.")
                else:
                    eq_r, eq_h = (1 + oos_r).cumprod(), (1 + oos_h).cumprod()
                    w1, w2, w3, w4 = st.columns(4)
                    w1.metric("Retorno OOS robô", f"{(eq_r.iloc[-1] - 1) * 100:.1f}%",
                              f"hold: {(eq_h.iloc[-1] - 1) * 100:.1f}%", delta_color="off")
                    w2.metric("Sharpe OOS", f"{_sharpe(oos_r):.2f}", f"hold: {_sharpe(oos_h):.2f}", delta_color="off")
                    w3.metric("DD OOS", f"{(eq_r / eq_r.cummax() - 1).min() * 100:.1f}%")
                    venceu = (tab_wf["Retorno Teste (%)"] > tab_wf["Hold Teste (%)"]).mean() * 100
                    w4.metric("Janelas em que bateu o hold", f"{venceu:.0f}%", f"{len(tab_wf)} janelas", delta_color="off")
                    fig_wf = go.Figure()
                    fig_wf.add_trace(go.Scatter(x=eq_h.index, y=(eq_h - 1) * 100, name="Hold (OOS)", line=dict(color="gray")))
                    fig_wf.add_trace(go.Scatter(x=eq_r.index, y=(eq_r - 1) * 100, name="Robô (OOS)", line=dict(color="green", width=2)))
                    for _, rw in tab_wf.iterrows():
                        fig_wf.add_vline(x=rw["Teste de"], line_dash="dot", line_color="lightgray")
                    fig_wf.update_layout(template="plotly_white", height=380, yaxis_title="Retorno acumulado (%)",
                                         title="Equity fora da amostra, janela a janela")
                    st.plotly_chart(fig_wf, **_W)
                    t = tab_wf.copy()
                    for c in ["Treino até", "Teste de", "Teste até"]:
                        t[c] = pd.to_datetime(t[c]).dt.strftime("%d/%m/%Y")
                    st.dataframe(t.style.format({"Sharpe Treino": "{:.2f}", "Retorno Teste (%)": "{:.1f}%",
                                                 "Hold Teste (%)": "{:.1f}%"})
                                 .background_gradient(subset=["Retorno Teste (%)"], cmap="RdYlGn", vmin=-30, vmax=30),
                                 hide_index=True, **_W)

            if diagnosticar:
                st.markdown("### 🩺 Diagnóstico do sinal")
                st.caption("O score prevê o retorno futuro? Correlação de cada componente com o retorno dos próximos "
                           "N dias e retorno médio por faixa de score. Componente com correlação negativa está "
                           "atrapalhando; faixa alta com retorno negativo = score esticado.")
                hz = st.session_state.get("diag_hz", 10)
                corr, by_bucket, n = signal_diagnostics(ind, hz)
                d1, d2 = st.columns([1, 1])
                d1.dataframe(corr.style.format({"Spearman": "{:+.3f}", "Pearson": "{:+.3f}"})
                             .background_gradient(subset=["Spearman"], cmap="RdYlGn", vmin=-0.2, vmax=0.2),
                             hide_index=True, **_W)
                d2.dataframe(by_bucket.style.format({f"Retorno médio {hz}d (%)": "{:+.2f}%", "% dias positivos": "{:.0f}%"})
                             .background_gradient(subset=[f"Retorno médio {hz}d (%)"], cmap="RdYlGn", vmin=-5, vmax=5),
                             hide_index=True, **_W)
                fig_b = px.bar(by_bucket, x="Faixa de Score", y=f"Retorno médio {hz}d (%)", template="plotly_white",
                               title=f"Retorno médio {hz} dias à frente por faixa de score (N={n})",
                               color=f"Retorno médio {hz}d (%)", color_continuous_scale="RdYlGn")
                fig_b.update_layout(height=320, coloraxis_showscale=False)
                st.plotly_chart(fig_b, **_W)
                pior = corr.sort_values("Spearman").iloc[0]
                melhor = corr[corr["Componente"] != "Score total"].sort_values("Spearman").iloc[-1]
                st.info(f"Melhor componente: **{melhor['Componente']}** ({melhor['Spearman']:+.3f}). "
                        f"Pior: **{pior['Componente']}** ({pior['Spearman']:+.3f}). "
                        "Correlações abaixo de ~0,05 em módulo são ruído; ajuste os pesos na sidebar e repita "
                        "em outros ativos e períodos antes de fixar.")

            if permutar:
                st.markdown("### 🎲 Teste de permutação")
                st.caption("Embaralha os retornos diários 200 vezes (mesma distribuição, sem estrutura temporal) e "
                           "roda o robô em cada série. Se o Sharpe real não supera ~95% dos embaralhados, "
                           "o resultado é indistinguível de sorte.")
                with st.spinner("Rodando 200 permutações..."):
                    sh_real, sh_perm, pval = permutation_test(h["Close"], P, gatilho_compra, gatilho_venda, fee_bt,
                                                              n_iter=200, skip_stretched=skip_stretched)
                p1, p2, p3 = st.columns(3)
                p1.metric("Sharpe real", f"{sh_real:.2f}")
                p2.metric("Sharpe embaralhado (mediana)", f"{np.median(sh_perm):.2f}",
                          f"p95: {np.percentile(sh_perm, 95):.2f}", delta_color="off")
                p3.metric("p-valor", f"{pval:.3f}", "há sinal" if pval < 0.05 else "sem evidência",
                          delta_color="normal" if pval < 0.05 else "inverse")
                fig_p = px.histogram(x=sh_perm, nbins=30, template="plotly_white",
                                     labels={"x": "Sharpe em séries embaralhadas"}, title="Distribuição nula")
                fig_p.add_vline(x=sh_real, line_color="green", line_width=3, annotation_text="real")
                fig_p.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_p, **_W)
                st.caption("O filtro de regime BTC não entra aqui (o BTC não é embaralhado junto); "
                           "o teste avalia o score e a regra de esticado.")
