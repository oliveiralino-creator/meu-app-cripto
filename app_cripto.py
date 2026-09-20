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
v3.3 (após BTC/ETH/SOL 5y):
  * Gatilhos são constantes: re-otimizar (rolante, expansivo) perdeu para 65/45 fixo nos três ativos.
    O rolante virou teste de robustez e mostra o gatilho fixo como referência no mesmo trecho.
  * Painel "Veredito por ativo": bateria completa (IS, permutação, OOS fixo, correlação) para vários
    ativos de uma vez. Resultado até aqui: SOL ✅ (p=0,007), ETH/BTC sem sinal (p≈0,17) — o motor
    funciona em ativos de beta alto; em BTC/ETH só reduz drawdown.
v3.5 (após 8 ativos):
  * 3 de 8 ativos com sinal real (SOL, DOGE, AVAX); vol alta é necessária mas não suficiente — o que importa é
    ter havido tendência longa, e isso não é previsível. Resposta: aba 🧺 Cesta (peso igual, robô em todos):
    5y +111% vs −55% do hold, DD −50% vs −85%, OOS Sharpe 0,72 vs 0,47.
  * Aba 📖 Guia explica cada variável, métrica, teste e filtro.
v3.6 — sentimento histórico como 5º componente (experimental, peso 0):
  * build_sentiment_history.py coleta manchetes datadas (GDELT) por ativo/semana e pontua (léxico e/ou Gemini).
  * O backtest lê sentimento_historico.csv sem look-ahead (semana vale a partir da segunda seguinte) e o botão
    "🗞️ Testar sentimento" compara com/sem, in-sample, OOS e permutação.
v3.7 — resultado do sentimento em SOL (tom diário GDELT, build_sentiment_timeline.py):
  * Como componente do score PIORA (atrasa saídas): peso fica 0.
  * Como FILTRO DE ENTRADA (sentimento ≥ 50) melhora: Sharpe OOS 1,05 → 1,18, metade das entradas.
    Opção na sidebar ("Confirmar entradas com sentimento"), desligada por padrão até validar em mais ativos.
  * Volume de cobertura como filtro contrarian: descartado (bloqueia os rompimentos que pagam).
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
    w_sent=0.0,         # sentimento histórico de notícias (5º componente) — 0 até ser validado
    mom_extreme=15.0,   # acima deste % no período, o momentum passa a ser penalizado (esticado)
    rsi_extreme=75.0,   # RSI acima disto marca o ativo como esticado
)
SENT_FILE = "sentimento_historico.csv"


@st.cache_data(ttl=3600, show_spinner=False)
def load_sentiment_history(path: str = SENT_FILE, prefer_ai: bool = True):
    """
    Lê o CSV gerado por build_sentiment_history.py e devolve {ativo: Series diária 0–100}.
    Cada linha é uma semana (segunda a domingo). Para não haver look-ahead, o score da semana passa a valer
    a partir da segunda-feira SEGUINTE e é mantido até a próxima atualização (ffill).
    """
    if not os.path.exists(path):
        return {}, None
    try:
        df = pd.read_csv(path)
        df["data"] = pd.to_datetime(df["data"], format="mixed", utc=True).dt.tz_localize(None).dt.normalize()
    except Exception as e:
        return {}, f"não foi possível ler {path}: {e}"
    out = {}
    if "tone" in df.columns:
        # ---- formato DIÁRIO (build_sentiment_timeline.py): tom do GDELT, escala ~ -10..+10 ----
        for a, g in df.groupby("ativo"):
            g = g.sort_values("data").dropna(subset=["tone"])
            s = pd.Series(g["tone"].values, index=pd.to_datetime(g["data"])).astype(float)
            s = s[~s.index.duplicated()].resample("D").mean()
            # suaviza (7 dias) e normaliza pela própria história do ativo, só com o passado (rolling z-score)
            sm = s.rolling(7, min_periods=3).mean()
            mu = sm.rolling(365, min_periods=60).mean()
            sd = sm.rolling(365, min_periods=60).std()
            z = ((sm - mu) / sd.replace(0, np.nan)).clip(-3, 3)
            score = (50 + z * (50 / 3)).shift(1)               # tom de hoje só é conhecido amanhã
            out[a] = {"serie": score, "fonte": "gdelt_tone", "semanas": int(score.notna().sum() // 7),
                      "de": g["data"].min().date(), "ate": g["data"].max().date()}
        return out, None
    for a, g in df.groupby("ativo"):
        # ---- formato SEMANAL (build_sentiment_history.py) ----
        g = g.sort_values("data")
        col = "sent_ia" if prefer_ai and "sent_ia" in g and g["sent_ia"].notna().sum() >= 0.5 * len(g) else "sent_lexico"
        s = pd.Series(g[col].values, index=pd.to_datetime(g["data"]) + pd.Timedelta(days=7)).astype(float)
        s = s.where(g["n_manchetes"].values >= 3)             # semanas com < 3 manchetes → sem informação
        daily = s.resample("D").last().ffill(limit=21)         # vale por até 3 semanas sem notícia nova
        out[a] = {"serie": daily, "fonte": col, "semanas": int(g["n_manchetes"].ge(3).sum()),
                  "de": g["data"].min().date(), "ate": g["data"].max().date()}
    return out, None
DEFAULT_SETTINGS = dict(use_regime=False, regime_ma=200, skip_stretched=False, buy_thr=65, sell_thr=45, fee=0.10,
                        sent_gate=False, sent_gate_thr=50)
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


def compute_indicators(close: pd.Series, volume: pd.Series | None = None, p: dict | None = None,
                       sent: pd.Series | None = None) -> pd.DataFrame:
    """
    Calcula indicadores e o Score técnico (0–100) para QUALQUER série de preços regular
    (diária no backtest e na watchlist; horária no sparkline do radar).
    Todos os indicadores olham apenas para trás → sem look-ahead.
    sent: série diária 0–100 de sentimento histórico (opcional; entra com peso p["w_sent"]).
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

    score = (p["w_rsi"] * df["S_RSI"] + p["w_trend"] * df["S_TREND"]
             + p["w_mom"] * df["S_MOM"] + p["w_vol"] * df["S_VOL"])
    wtot = p["w_rsi"] + p["w_trend"] + p["w_mom"] + p["w_vol"]
    if sent is not None and p.get("w_sent", 0) > 0:
        s = pd.Series(sent).reindex(df.index).ffill(limit=21) if isinstance(df.index, pd.DatetimeIndex) else None
        if s is not None and s.notna().any():
            df["S_SENT"] = s.clip(0, 100)
            # dia sem sentimento: o componente sai do cálculo e os demais pesos são renormalizados
            # (preencher com 50 diluía o score com uma constante e bloqueava os gatilhos)
            has = df["S_SENT"].notna()
            score = np.where(has, score + p["w_sent"] * df["S_SENT"].fillna(0), score)
            wtot = np.where(has, wtot + p["w_sent"], wtot)
    df["Score"] = score / np.where(wtot == 0, 1.0, wtot)
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
                 periods_per_year: int = 365, regime: pd.Series | None = None, skip_stretched: bool = False,
                 entry_gate: pd.Series | None = None):
    """
    Sinal gerado no fechamento de t, exposição vale a partir de t+1 (shift). Taxa cobrada a cada troca de posição.
    regime: série booleana (True = pode estar comprado); fora do regime a posição é zerada.
    skip_stretched: não abre compra quando o ativo está esticado (mas mantém posição já aberta).
    entry_gate: série booleana — False bloqueia NOVAS compras naquele dia (a saída não muda). Dias ausentes = True.
    Retorna (DataFrame com equity, dict de métricas, DataFrame de trades).
    """
    df = ind.copy()
    buy = df["Score"] >= buy_thr
    if skip_stretched and "Esticado" in df:
        buy = buy & ~df["Esticado"].fillna(False).astype(bool)
    if entry_gate is not None:
        buy = buy & entry_gate.reindex(df.index).fillna(True).astype(bool)
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
             "S_MOM": "Momentum", "S_VOL": "Risco (baixa vol)", "S_SENT": "Sentimento (notícias)"}
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


def asset_verdict(close: pd.Series, p: dict, buy_thr=65, sell_thr=45, fee_pct=0.1, oos_start=252, n_perm=100, sent=None,
                  entry_gate=None):
    """
    Bateria de validação de um ativo com gatilhos FIXOS:
      in-sample completo, permutação (p-valor), trecho OOS (a partir de oos_start, sem nenhuma otimização)
      e correlação do score com o retorno de 20 dias. Devolve dict com métricas e um veredito.
    """
    ind = compute_indicators(close, None, p, sent).dropna(subset=["Score"])
    if len(ind) < oos_start + 120:
        return {"erro": f"histórico curto ({len(ind)} dias)"}
    _, m_is, _ = run_backtest(ind, buy_thr, sell_thr, fee_pct, entry_gate=entry_gate)
    _, m_oos, _ = run_backtest(ind.iloc[oos_start:], buy_thr, sell_thr, fee_pct, entry_gate=entry_gate)
    corr, _, _ = signal_diagnostics(ind, 20)
    sh_real, perm, pval = permutation_test(close, p, buy_thr, sell_thr, fee_pct, n_iter=n_perm, sent=sent, entry_gate=entry_gate)
    vol = float(close.pct_change().std() * np.sqrt(365) * 100)
    sinal = pval < 0.05
    bate_hold = m_oos["sharpe_robo"] > m_oos["sharpe_hold"] and m_oos["ret_robo"] > m_oos["ret_hold"]
    protege = m_oos["dd_robo"] > m_oos["dd_hold"] and m_oos["sharpe_robo"] >= m_oos["sharpe_hold"] - 0.05
    if sinal and bate_hold:
        veredito = "✅ Sinal real — usar robô"
    elif protege:
        veredito = "🛡️ Só reduz drawdown — hold ou robô como freio"
    else:
        veredito = "❌ Sem vantagem — hold"
    return {"Vol anual (%)": vol, "Robô IS (%)": m_is["ret_robo"], "Hold IS (%)": m_is["ret_hold"],
            "Sharpe IS": m_is["sharpe_robo"], "DD robô (%)": m_is["dd_robo"], "DD hold (%)": m_is["dd_hold"],
            "Trades": m_is["n_trades"], "Corr 20d": float(corr.iloc[0]["Spearman"]),
            "p-valor": pval, "Robô OOS (%)": m_oos["ret_robo"], "Hold OOS (%)": m_oos["ret_hold"],
            "Sharpe OOS": m_oos["sharpe_robo"], "Sharpe hold OOS": m_oos["sharpe_hold"],
            "Veredito": veredito, "_dias": len(ind)}


def basket_backtest(hist_map: dict, p: dict, buy_thr=65, sell_thr=45, fee_pct=0.1, oos_start=252, sent_map=None,
                    gate_fn=None, **bt_kw):
    """
    Roda o robô em cada ativo e monta uma cesta de peso igual, rebalanceada diariamente.
    Retorna dict com séries de retorno (robô/hold), tabela por ativo, por ano e estado atual de cada ativo.
    """
    R, H, estado, contrib = {}, {}, [], []
    for t, h in hist_map.items():
        a = t.replace("-USD", "")
        ind = compute_indicators(h["Close"], h.get("Volume"), p, (sent_map or {}).get(a)).dropna(subset=["Score"])
        if len(ind) < 60:
            continue
        res, m, _ = run_backtest(ind, buy_thr, sell_thr, fee_pct, entry_gate=(gate_fn(a) if gate_fn else None), **bt_kw)
        R[a], H[a] = res["Ret_Robo"], res["Ret"]
        last = res.iloc[-1]
        dias_pos = int((res["Pos"].iloc[::-1] != last["Pos"]).values.argmax()) if (res["Pos"] != last["Pos"]).any() else len(res)
        estado.append({"Ativo": a, "Preço": float(last["Close"]), "Score": float(last["Score"]),
                       "Posição": "🟢 Comprado" if last["Pos"] == 1 else "⚪ Em caixa", "Há (dias)": dias_pos,
                       "Esticado": "⚠️" if bool(last["Esticado"]) else "", "RSI": float(last["RSI"]),
                       "Tendência": float(last["S_TREND"]), "Momentum": float(last["S_MOM"]), "Risco": float(last["S_VOL"])})
        contrib.append({"Ativo": a, "Robô (%)": m["ret_robo"], "Hold (%)": m["ret_hold"], "DD robô (%)": m["dd_robo"],
                        "DD hold (%)": m["dd_hold"], "Sharpe robô": m["sharpe_robo"], "Sharpe hold": m["sharpe_hold"],
                        "Trades": m["n_trades"], "Exposição (%)": m["exposure"]})
    if not R:
        return None
    R, H = pd.DataFrame(R).dropna(), pd.DataFrame(H).dropna()
    rr, hh = R.mean(axis=1), H.mean(axis=1)

    def _m(r):
        e = (1 + r).cumprod()
        return {"ret": (e.iloc[-1] - 1) * 100, "dd": (e / e.cummax() - 1).min() * 100, "sharpe": _sharpe(r), "eq": e}
    out = {"robo": _m(rr), "hold": _m(hh), "n": len(R.columns), "dias": len(rr)}
    if len(rr) > oos_start + 60:
        out["robo_oos"], out["hold_oos"] = _m(rr.iloc[oos_start:]), _m(hh.iloc[oos_start:])
    # contribuição: retorno médio diário de cada ativo dentro da cesta (já dividido por N)
    ct = pd.DataFrame(contrib)
    ct["Contribuição p/ cesta (pp)"] = [((1 + R[a]).prod() - 1) * 100 / len(R.columns) for a in ct["Ativo"]]
    out["por_ativo"] = ct.sort_values("Contribuição p/ cesta (pp)", ascending=False)
    yr = pd.DataFrame({"Robô (%)": rr.groupby(rr.index.year).apply(lambda s: ((1 + s).prod() - 1) * 100),
                       "Hold (%)": hh.groupby(hh.index.year).apply(lambda s: ((1 + s).prod() - 1) * 100)})
    yr["Exposição média (%)"] = pd.DataFrame({a: (R[a] != 0).astype(float) for a in R}).mean(axis=1).groupby(rr.index.year).mean() * 100
    out["por_ano"] = yr
    out["estado"] = pd.DataFrame(estado).sort_values("Score", ascending=False)
    out["exposicao_hoje"] = float((out["estado"]["Posição"].str.startswith("🟢")).mean() * 100)
    return out


def permutation_test(close: pd.Series, p: dict, buy_thr, sell_thr, fee_pct, n_iter=200, seed=0, sent=None, **bt_kw):
    """
    Embaralha os retornos diários (destrói a estrutura temporal, preserva a distribuição), recalcula
    indicadores e roda o robô. Retorna (Sharpe real, array de Sharpes embaralhados, p-valor).
    """
    rng = np.random.default_rng(seed)
    ind = compute_indicators(close, None, p, sent).dropna(subset=["Score"])
    _, m_real, _ = run_backtest(ind, buy_thr, sell_thr, fee_pct, **bt_kw)
    rets = close.pct_change().dropna().values
    sharpes = []
    for _ in range(n_iter):
        shuffled = pd.Series(close.iloc[0] * np.cumprod(1 + rng.permutation(rets)), index=close.index[1:])
        ind_s = compute_indicators(shuffled, None, p, sent).dropna(subset=["Score"])
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
         st.slider("Risco (volatilidade)", 0.0, 1.0, float(P["w_vol"]), 0.05),
         st.slider("Sentimento histórico (notícias)", 0.0, 1.0, float(P.get("w_sent", 0.0)), 0.05,
                   help="Só tem efeito no backtest e se existir sentimento_historico.csv (gerado por "
                        "build_sentiment_history.py). Padrão 0 até ser validado.")]
    tot = sum(w) or 1.0
    P["w_rsi"], P["w_trend"], P["w_mom"], P["w_vol"], P["w_sent"] = [round(x / tot, 4) for x in w]
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

with st.sidebar.expander("🗞️ Sentimento de notícias (histórico)", expanded=False):
    sent_gate_on = st.toggle("Confirmar entradas com sentimento", value=bool(S0.get("sent_gate", False)),
                             help="Só abre compra quando o sentimento histórico do ativo está acima do limiar; a saída "
                                  "não muda. Em SOL (2021–26) subiu o Sharpe fora da amostra de 1,12 para 1,34 e cortou "
                                  "as entradas quase pela metade. Validado em UM ativo — desligado por padrão. "
                                  "Requer sentimento_historico.csv.")
    sent_gate_thr = st.slider("Limiar do sentimento", 30, 70, int(S0.get("sent_gate_thr", 50)), 5)
    st.caption("Como componente do score (peso na seção acima) o sentimento PIOROU o resultado — ele atrasa saídas.")

st.sidebar.divider()
st.sidebar.caption(("📂 Parâmetros carregados de `parametros.json`." if params_from_file
                    else "Usando padrões validados (SOL 5y).") +
                   " Salve para que sobrevivam ao recarregar o app.")
if st.sidebar.button("💾 Salvar parâmetros e filtros"):
    S_now = {"use_regime": use_regime, "regime_ma": regime_ma, "skip_stretched": skip_stretched,
             "sent_gate": sent_gate_on, "sent_gate_thr": sent_gate_thr,
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
SENT_HIST, sent_err = load_sentiment_history()
SENT_MAP = {a: v["serie"] for a, v in SENT_HIST.items()}


def sent_gate_for(ativo):
    """Série booleana de permissão de entrada por sentimento (None se desligado ou sem dados para o ativo)."""
    if not sent_gate_on or ativo not in SENT_MAP:
        return None
    return SENT_MAP[ativo] >= sent_gate_thr

# =====================================================================
# 7. ABAS
# =====================================================================
tab1, tab2, tab3, tab5, tab4 = st.tabs(["📊 Radar de Mercado", "💼 Simulador de Carteira", "⏪ Backtesting",
                                        "🧺 Cesta", "📖 Guia"])

# Textos de ajuda reutilizados nas tabelas (versão curta; a aba Guia tem a completa)
HELP_RADAR = """
| Coluna | O que é | Como ler |
|---|---|---|
| **RSI** | Índice de força relativa (14 dias): mede se o ativo subiu "demais" ou caiu "demais" recentemente. 0–100. | < 30 sobrevendido, > 70 sobrecomprado. |
| **Tendência** | 0–100. Soma de três checagens: média rápida acima da lenta (35 pts), MACD acima do sinal (35), preço acima da média de 50 dias (30). | 100 = todas as três apontam alta; 0 = nenhuma. |
| **Momentum** | 0–100. Variação dos últimos 10 dias (−10% → 0, +10% → 100). Acima de +15% o valor **decai**: alta rápida demais = esticado. | Alto é bom até certo ponto; ⚠️ Esticado avisa quando passou. |
| **Risco** | 0–100. Quanto **menor** a volatilidade dos últimos 20 dias, maior o valor. | Alto = ativo calmo; baixo = ativo nervoso. |
| **Sentimento** | 0–100 a partir das manchetes recentes do ativo (ou gerais, se houver poucas). 50 = neutro. | Pesa pouco no score final (15%). |
| **Score Técnico** | Média ponderada de RSI, Tendência, Momentum e Risco com os pesos da sidebar. | É o número validado pelo backtesting. |
| **Score Final** | 70% Score Técnico + 15% Sentimento + 15% Fear & Greed invertido (medo extremo soma pontos). | Base da coluna Decisão. |
| **Esticado ⚠️** | Momentum > 15% em 10 dias ou RSI > 75. | Aviso: alta forte recente; historicamente o retorno seguinte é pior. |
| **Decisão** | ≥ 80 Compra Forte · 60–79 Compra Média · 51–59 Compra Fraca · 45–50 Neutro · 41–44 Venda Fraca · 21–40 Venda Média · ≤ 20 Venda Forte. | No backtest, o robô compra com score ≥ 65 e vende com ≤ 45. |
"""
HELP_BACKTEST = """
| Métrica | O que é | Como ler |
|---|---|---|
| **Retorno robô / hold** | Quanto rendeu seguir o robô vs. comprar no início e segurar. | O robô precisa vencer o hold **e** com menos risco para valer a pena. |
| **Drawdown máx.** | Maior queda do pico ao vale ao longo do período. | −50% significa que em algum momento você viu metade do capital sumir. |
| **Sharpe (anual.)** | Retorno por unidade de risco. | < 0,5 fraco · 0,5–1 razoável · > 1 bom. Compare sempre com o Sharpe do hold. |
| **CAGR** | Retorno composto por ano. | Permite comparar períodos de tamanhos diferentes. |
| **Trades / win rate** | Nº de operações completas e % delas com lucro. | Seguidores de tendência têm win rate baixo (30–45%) e ganhos grandes: normal. |
| **Payoff** | Ganho médio dos trades vencedores ÷ perda média dos perdedores. | > 2 compensa win rate baixo. |
| **Exposição** | % do tempo em que o robô esteve comprado. | O restante do tempo o capital ficou parado (em caixa). |
| **p-valor (permutação)** | Probabilidade de obter o mesmo Sharpe por sorte, em séries embaralhadas. | < 0,05 = há sinal real. > 0,10 = indistinguível de sorte. |
| **OOS / fora da amostra** | Resultado em dados que não foram usados para escolher nada. | É o único número que aproxima o que esperar no futuro. |
| **Corr 20d** | Correlação (Spearman) entre o score de hoje e o retorno dos 20 dias seguintes. | > 0,08 já é relevante em finanças; ≈ 0 = o score não antecipa nada. |
"""

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

            with st.expander("❓ O que significa cada coluna"):
                st.markdown(HELP_RADAR)
                st.caption("Explicação completa, com exemplos, na aba 📖 Guia.")

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
                                              ("não comprar esticado", skip_stretched),
                                              (f"entrada só com sentimento ≥ {sent_gate_thr}", sent_gate_on)] if on) or "nenhum"
    st.caption(f"Filtros ativos (sidebar): **{filtros_txt}**")

    if sent_err:
        st.warning(sent_err)
    sent_bt = SENT_MAP.get(ativo_bt)
    if ativo_bt in SENT_HIST:
        i = SENT_HIST[ativo_bt]
        st.caption(f"🗞️ Sentimento histórico disponível para {ativo_bt}: {i['semanas']} semanas com notícias "
                   f"({i['de']} → {i['ate']}), fonte `{i['fonte']}`. Peso atual no score: {P.get('w_sent', 0):.2f}.")
    elif SENT_HIST:
        st.caption(f"🗞️ Sentimento histórico existe, mas não para {ativo_bt}. Ativos cobertos: "
                   + ", ".join(sorted(SENT_HIST)))
    else:
        st.caption("🗞️ Sem sentimento histórico. Gere com `python build_sentiment_history.py --ativos SOL,BTC,...` "
                   "e coloque o CSV na pasta do app para testar o sentimento como componente do score.")

    b1, b2, b3, b4, b5, b6 = st.columns(6)
    rodar = b1.button("▶️ Rodar simulação", disabled=gatilho_venda >= gatilho_compra)
    otimizar = b2.button("🧪 Walk-forward simples")
    rolante = b3.button("🔁 Walk-forward rolante")
    diagnosticar = b4.button("🩺 Diagnóstico do sinal")
    permutar = b5.button("🎲 Teste de permutação")
    testar_sent = b6.button("🗞️ Testar sentimento", disabled=sent_bt is None)

    if rodar or otimizar or rolante or diagnosticar or permutar or testar_sent:
        ticker = f"{ativo_bt}-USD"
        with st.spinner(f"Baixando {ticker}..."):
            hist_map, missing, err_yf = get_daily_history((ticker,), periodo_bt)
        if ticker not in hist_map:
            st.error(f"Sem dados para {ticker}" + (f": {err_yf}" if err_yf else " (ticker inexistente no yfinance?)"))
        else:
            h = hist_map[ticker]
            ind = compute_indicators(h["Close"], h.get("Volume"), P, sent_bt).dropna(subset=["Score"])
            BT_KW = {**BT_KW, "entry_gate": sent_gate_for(ativo_bt)}

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
                with st.expander("❓ O que significa cada métrica"):
                    st.markdown(HELP_BACKTEST)

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
                st.markdown("### 🔁 Walk-forward rolante — teste de robustez (100% fora da amostra)")
                st.caption("Treina numa janela, escolhe os gatilhos, aplica no trimestre seguinte e avança. "
                           "Em BTC, ETH e SOL 5y, re-otimizar gatilhos a cada trimestre PERDEU para o gatilho fixo "
                           "65/45 — por isso a linha tracejada abaixo mostra o gatilho fixo no mesmo trecho. "
                           "Use isto para checar robustez, não para escolher gatilhos.")
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
                    # referência: gatilho fixo da tela, aplicado ao mesmo trecho OOS, sem otimização alguma
                    res_fix, m_fix, _ = run_backtest(ind.loc[oos_r.index[0]:oos_r.index[-1]], gatilho_compra,
                                                     gatilho_venda, fee_bt, **BT_KW)
                    w1, w2, w3, w4 = st.columns(4)
                    w1.metric("Retorno OOS (re-otimizado)", f"{(eq_r.iloc[-1] - 1) * 100:.1f}%",
                              f"hold: {(eq_h.iloc[-1] - 1) * 100:.1f}%", delta_color="off")
                    w2.metric("Sharpe OOS (re-otimizado)", f"{_sharpe(oos_r):.2f}", f"hold: {_sharpe(oos_h):.2f}", delta_color="off")
                    w3.metric(f"Gatilho fixo {gatilho_compra}/{gatilho_venda} no mesmo trecho",
                              f"{m_fix['ret_robo']:.1f}%", f"Sharpe {m_fix['sharpe_robo']:.2f} · DD {m_fix['dd_robo']:.0f}%",
                              delta_color="off")
                    venceu = (tab_wf["Retorno Teste (%)"] > tab_wf["Hold Teste (%)"]).mean() * 100
                    w4.metric("Janelas em que bateu o hold", f"{venceu:.0f}%", f"{len(tab_wf)} janelas", delta_color="off")
                    if m_fix["sharpe_robo"] > _sharpe(oos_r):
                        st.info("O gatilho fixo superou a re-otimização neste ativo — comportamento esperado; "
                                "trate os gatilhos como constantes.")
                    fig_wf = go.Figure()
                    fig_wf.add_trace(go.Scatter(x=eq_h.index, y=(eq_h - 1) * 100, name="Hold (OOS)", line=dict(color="gray")))
                    fig_wf.add_trace(go.Scatter(x=eq_r.index, y=(eq_r - 1) * 100, name="Robô re-otimizado (OOS)",
                                                line=dict(color="green", width=2)))
                    fig_wf.add_trace(go.Scatter(x=res_fix.index, y=(res_fix["Eq_Robo"] - 1) * 100,
                                                name=f"Robô gatilho fixo {gatilho_compra}/{gatilho_venda}",
                                                line=dict(color="darkgreen", width=2, dash="dash")))
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
                                                              n_iter=200, sent=sent_bt, skip_stretched=skip_stretched)
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

            if testar_sent:
                st.markdown("### 🗞️ O sentimento das notícias ajuda o robô?")
                st.caption("Mesma série, mesmos gatilhos: score sem sentimento vs. score com sentimento (peso 0,25, os "
                           "demais reescalados). O sentimento de cada semana só vale a partir da segunda-feira seguinte "
                           "(sem look-ahead). Permutação com 150 embaralhamentos em cada caso.")
                P0s = {**P, "w_sent": 0.0}
                P1s = {**P, "w_sent": 0.25}
                ind0 = compute_indicators(h["Close"], None, P0s).dropna(subset=["Score"])
                ind1 = compute_indicators(h["Close"], None, P1s, sent_bt).dropna(subset=["Score"])
                if "S_SENT" not in ind1:
                    st.error("O sentimento não cobre o período deste backtest.")
                else:
                    cobertura = float(sent_bt.reindex(ind1.index).notna().mean() * 100)
                    st.caption(f"Cobertura do sentimento no período: {cobertura:.0f}% dos dias.")
                    KW0 = {k: v for k, v in BT_KW.items() if k != "entry_gate"}     # comparação limpa, sem o gate
                    gate = sent_bt >= sent_gate_thr
                    with st.spinner("Rodando: sem sentimento · como componente · como filtro de entrada..."):
                        _, m0, _ = run_backtest(ind0, gatilho_compra, gatilho_venda, fee_bt, **KW0)
                        _, m1, _ = run_backtest(ind1, gatilho_compra, gatilho_venda, fee_bt, **KW0)
                        _, m2, _ = run_backtest(ind0, gatilho_compra, gatilho_venda, fee_bt, entry_gate=gate, **KW0)
                        _, o0, _ = run_backtest(ind0.iloc[252:], gatilho_compra, gatilho_venda, fee_bt, **KW0)
                        _, o1, _ = run_backtest(ind1.iloc[252:], gatilho_compra, gatilho_venda, fee_bt, **KW0)
                        _, o2, _ = run_backtest(ind0.iloc[252:], gatilho_compra, gatilho_venda, fee_bt, entry_gate=gate, **KW0)
                        _, _, p0 = permutation_test(h["Close"], P0s, gatilho_compra, gatilho_venda, fee_bt, n_iter=120, **KW0)
                        _, _, p1 = permutation_test(h["Close"], P1s, gatilho_compra, gatilho_venda, fee_bt, n_iter=120,
                                                    sent=sent_bt, **KW0)
                        _, _, p2 = permutation_test(h["Close"], P0s, gatilho_compra, gatilho_venda, fee_bt, n_iter=120,
                                                    entry_gate=gate, **KW0)
                        corr1, by1, _ = signal_diagnostics(ind1, 20)
                    comp = pd.DataFrame({
                        "Sem sentimento": [m0["ret_robo"], m0["dd_robo"], m0["sharpe_robo"], m0["n_trades"], o0["ret_robo"], o0["sharpe_robo"], p0],
                        "No score (peso 0,25)": [m1["ret_robo"], m1["dd_robo"], m1["sharpe_robo"], m1["n_trades"], o1["ret_robo"], o1["sharpe_robo"], p1],
                        f"Filtro de entrada (≥ {sent_gate_thr})": [m2["ret_robo"], m2["dd_robo"], m2["sharpe_robo"], m2["n_trades"], o2["ret_robo"], o2["sharpe_robo"], p2],
                    }, index=["Retorno (%)", "Drawdown (%)", "Sharpe", "Trades", "Retorno OOS (%)", "Sharpe OOS", "p-valor"])
                    st.dataframe(comp.style.format("{:.2f}"), **_W)
                    if o2["sharpe_robo"] > o0["sharpe_robo"] and p2 <= max(p0, 0.05):
                        st.success(f"Como **filtro de entrada** o sentimento melhorou o Sharpe fora da amostra "
                                   f"({o0['sharpe_robo']:.2f} → {o2['sharpe_robo']:.2f}). Ative em '🗞️ Sentimento' na sidebar "
                                   "se isso se repetir em outros ativos.")
                    rs = corr1[corr1["Componente"] == "Sentimento (notícias)"]
                    if len(rs):
                        cs = float(rs["Spearman"].iloc[0])
                        st.metric("Correlação do sentimento com o retorno de 20 dias", f"{cs:+.3f}",
                                  "sinal relevante" if abs(cs) >= 0.05 else "ruído", delta_color="normal" if cs >= 0.05 else "inverse")
                    if (o1["sharpe_robo"] > o0["sharpe_robo"]) and (p1 <= p0):
                        st.info("Como componente do score também melhorou — incomum; confirme em outros ativos antes de subir o peso.")
                    else:
                        st.warning("Como **componente do score** o sentimento não ajudou (em SOL piorou: atrasa as saídas). "
                                   "Mantenha o peso em 0.")
                    st.dataframe(corr1.style.format({"Spearman": "{:+.3f}", "Pearson": "{:+.3f}"}), hide_index=True, **_W)

    # ------------------------------------------------------------------
    # VEREDITO POR ATIVO — bateria completa com gatilho fixo, vários ativos de uma vez
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### 🏁 Veredito por ativo (gatilho fixo, 5 anos)")
    st.caption("Para cada ativo: backtest completo, teste de permutação, trecho fora da amostra (a partir do dia 252, "
               "sem otimização) e correlação do score com o retorno de 20 dias. "
               "✅ = p<0,05 e bate o hold fora da amostra · 🛡️ = sem sinal, mas reduz drawdown com Sharpe ≥ hold · "
               "❌ = sem vantagem. Regra observada: quanto maior a volatilidade do ativo, mais o motor funciona "
               "(SOL ✅, ETH/BTC 🛡️).")
    v1, v2, v3 = st.columns([3, 1, 1])
    sugeridos = sorted(set(watchlist) | {"BTC", "ETH", "SOL"})
    ativos_ver = v1.multiselect("Ativos:", options=sorted(set(sugeridos) | set(lista_ativos)), default=sugeridos,
                                max_selections=12)
    extra = v2.text_input("Extras (vírgula):", value="", placeholder="AVAX, LINK, DOGE")
    n_perm = v3.select_slider("Permutações", options=[50, 100, 200], value=100)
    ativos_ver = list(dict.fromkeys(ativos_ver + [a.strip().upper() for a in extra.split(",") if a.strip()]))
    if st.button("🏁 Rodar veredito", disabled=not ativos_ver):
        tick = tuple(f"{a}-USD" for a in ativos_ver)
        with st.spinner("Baixando 5 anos..."):
            hist_v, miss_v, err_v = get_daily_history(tick, "5y")
        if miss_v:
            st.caption("Sem histórico: " + ", ".join(m.replace("-USD", "") for m in miss_v))
        linhas, barra = [], st.progress(0.0)
        for i, a in enumerate(ativos_ver):
            t = f"{a}-USD"
            if t in hist_v:
                r = asset_verdict(hist_v[t]["Close"], P, gatilho_compra, gatilho_venda, fee_bt, n_perm=n_perm,
                                  sent=SENT_MAP.get(a), entry_gate=sent_gate_for(a))
                linhas.append({"Ativo": a, **({"Veredito": r["erro"]} if "erro" in r else r)})
            barra.progress((i + 1) / len(ativos_ver), text=f"{a} ({i + 1}/{len(ativos_ver)})")
        barra.empty()
        if linhas:
            dv = pd.DataFrame(linhas)
            cols_v = ["Ativo", "Veredito", "Vol anual (%)", "p-valor", "Corr 20d", "Robô OOS (%)", "Hold OOS (%)",
                      "Sharpe OOS", "Sharpe hold OOS", "Robô IS (%)", "Hold IS (%)", "DD robô (%)", "DD hold (%)", "Trades"]
            dv = dv[[c for c in cols_v if c in dv]].sort_values("p-valor") if "p-valor" in dv else dv
            st.session_state["veredito"] = dv
    if "veredito" in st.session_state:
        dv = st.session_state["veredito"]
        fmt = {"Vol anual (%)": "{:.0f}%", "p-valor": "{:.3f}", "Corr 20d": "{:+.3f}", "Robô OOS (%)": "{:.0f}%",
               "Hold OOS (%)": "{:.0f}%", "Sharpe OOS": "{:.2f}", "Sharpe hold OOS": "{:.2f}", "Robô IS (%)": "{:.0f}%",
               "Hold IS (%)": "{:.0f}%", "DD robô (%)": "{:.0f}%", "DD hold (%)": "{:.0f}%"}
        sty = dv.style.format({k: v for k, v in fmt.items() if k in dv})
        if "p-valor" in dv:
            sty = sty.background_gradient(subset=["p-valor"], cmap="RdYlGn_r", vmin=0, vmax=0.3)
        st.dataframe(sty, hide_index=True, **_W)
        if "Vol anual (%)" in dv and len(dv) >= 3:
            fig_v = px.scatter(dv, x="Vol anual (%)", y="p-valor", text="Ativo", template="plotly_white",
                               title="Sinal × volatilidade (abaixo da linha = sinal estatisticamente real)")
            fig_v.add_hline(y=0.05, line_dash="dash", line_color="green")
            fig_v.update_traces(textposition="top center"); fig_v.update_layout(height=340)
            st.plotly_chart(fig_v, **_W)
        st.download_button("📥 Baixar veredito (CSV)", dv.to_csv(index=False).encode("utf-8"), "veredito_ativos.csv", "text/csv")

# ---------------------------------------------------------------------
# ABA 5 — CESTA
# ---------------------------------------------------------------------
with tab5:
    st.subheader("🧺 Cesta — o robô aplicado a vários ativos ao mesmo tempo")
    st.caption("Conclusão dos testes: não dá para saber de antemão qual altcoin vai ter tendência, mas numa cesta os "
               "acertos grandes de uns pagam as perdas pequenas dos outros. Peso igual entre os ativos, rebalanceado "
               "diariamente; capital dos ativos em caixa fica parado. Mesmo motor e gatilhos da aba Backtesting.")
    CESTA_PADRAO = ["SOL", "AVAX", "DOGE", "ETH", "BTC", "LINK", "ADA", "DOT"]
    c1, c2, c3 = st.columns([3, 1, 1])
    opcoes_cesta = sorted(set(CESTA_PADRAO) | set(watchlist) | set(lista_ativos))
    ativos_cesta = c1.multiselect("Ativos da cesta:", options=opcoes_cesta,
                                  default=[a for a in CESTA_PADRAO if a in opcoes_cesta], max_selections=15)
    extra_c = c2.text_input("Extras (vírgula):", value="", key="cesta_extra", placeholder="NEAR, SUI")
    periodo_c = c3.selectbox("Período:", ["5y", "2y", "1y", "max"], key="cesta_periodo")
    ativos_cesta = list(dict.fromkeys(ativos_cesta + [a.strip().upper() for a in extra_c.split(",") if a.strip()]))
    st.caption(f"Gatilhos {S0['buy_thr']}/{S0['sell_thr']} · taxa {S0['fee']:.2f}% · filtros: "
               + (", ".join(f for f, on in [("regime BTC", use_regime and REGIME is not None),
                                              ("não comprar esticado", skip_stretched)] if on) or "nenhum")
               + " — altere na aba Backtesting / sidebar.")

    if st.button("🧺 Rodar cesta", disabled=not ativos_cesta):
        with st.spinner(f"Baixando {len(ativos_cesta)} ativos..."):
            hist_c, miss_c, err_c = get_daily_history(tuple(f"{a}-USD" for a in ativos_cesta), periodo_c)
        if miss_c:
            st.caption("Sem histórico: " + ", ".join(m.replace("-USD", "") for m in miss_c))
        with st.spinner("Rodando o robô em cada ativo..."):
            st.session_state["cesta"] = basket_backtest(hist_c, P, int(S0["buy_thr"]), int(S0["sell_thr"]),
                                                        float(S0["fee"]), sent_map=SENT_MAP, gate_fn=sent_gate_for, **BT_KW)
        if st.session_state["cesta"] is None:
            st.error("Nenhum ativo com histórico suficiente.")

    B = st.session_state.get("cesta")
    if B:
        # ---------- hoje ----------
        st.markdown("### Hoje")
        h1, h2, h3, h4 = st.columns(4)
        n_comp = int(B["estado"]["Posição"].str.startswith("🟢").sum())
        h1.metric("Exposição da cesta", f"{B['exposicao_hoje']:.0f}%", f"{n_comp} de {B['n']} ativos comprados",
                  delta_color="off")
        h2.metric("Score médio", f"{B['estado']['Score'].mean():.0f}")
        h3.metric("Esticados", f"{(B['estado']['Esticado'] == '⚠️').sum()}")
        h4.metric("Dias no histórico", f"{B['dias']}")
        st.dataframe(B["estado"].style.format({"Preço": fmt_price, "Score": "{:.0f}", "RSI": "{:.0f}",
                                               "Tendência": "{:.0f}", "Momentum": "{:.0f}", "Risco": "{:.0f}"})
                     .background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100), hide_index=True, **_W)
        st.caption("'Há (dias)' = há quantos dias o robô está nesse estado. Score ≥ gatilho de compra com posição "
                   "'Em caixa' significa que a compra vale a partir do próximo fechamento.")

        # ---------- desempenho ----------
        st.markdown("### Desempenho da cesta")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Retorno robô", f"{B['robo']['ret']:.0f}%", f"hold: {B['hold']['ret']:.0f}%", delta_color="off")
        k2.metric("Drawdown máx.", f"{B['robo']['dd']:.0f}%", f"hold: {B['hold']['dd']:.0f}%", delta_color="off")
        k3.metric("Sharpe", f"{B['robo']['sharpe']:.2f}", f"hold: {B['hold']['sharpe']:.2f}", delta_color="off")
        if "robo_oos" in B:
            k4.metric("Fora da amostra (dia 252+)", f"{B['robo_oos']['ret']:.0f}% · Sharpe {B['robo_oos']['sharpe']:.2f}",
                      f"hold: {B['hold_oos']['ret']:.0f}% · Sharpe {B['hold_oos']['sharpe']:.2f}", delta_color="off")
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=B["hold"]["eq"].index, y=(B["hold"]["eq"] - 1) * 100, name="Cesta hold",
                                   line=dict(color="gray")))
        fig_c.add_trace(go.Scatter(x=B["robo"]["eq"].index, y=(B["robo"]["eq"] - 1) * 100, name="Cesta robô",
                                   line=dict(color="green", width=2)))
        if "robo_oos" in B:
            fig_c.add_vline(x=B["robo_oos"]["eq"].index[0], line_dash="dash", line_color="orange",
                            annotation_text="fora da amostra →", annotation_position="top left")
        fig_c.update_layout(template="plotly_white", height=400, yaxis_title="Retorno acumulado (%)",
                            title=f"Cesta de {B['n']} ativos — robô vs. hold")
        st.plotly_chart(fig_c, **_W)

        cA, cB = st.columns([1, 1])
        with cA:
            st.markdown("**Por ano**")
            st.dataframe(B["por_ano"].style.format("{:.0f}%")
                         .background_gradient(subset=["Robô (%)"], cmap="RdYlGn", vmin=-60, vmax=60), **_W)
        with cB:
            st.markdown("**Contribuição por ativo** (pp do retorno da cesta)")
            st.dataframe(B["por_ativo"].style.format({"Robô (%)": "{:.0f}%", "Hold (%)": "{:.0f}%", "DD robô (%)": "{:.0f}%",
                                                      "DD hold (%)": "{:.0f}%", "Sharpe robô": "{:.2f}", "Sharpe hold": "{:.2f}",
                                                      "Exposição (%)": "{:.0f}%", "Contribuição p/ cesta (pp)": "{:+.1f}"})
                         .background_gradient(subset=["Contribuição p/ cesta (pp)"], cmap="RdYlGn", vmin=-30, vmax=30),
                         hide_index=True, **_W)
        piores = B["por_ativo"][B["por_ativo"]["Sharpe robô"] < B["por_ativo"]["Sharpe hold"]]["Ativo"].tolist()
        if piores:
            st.info(f"Ativos em que o robô ficou atrás do hold neste período: **{', '.join(piores)}**. "
                    "Isso é esperado em parte da cesta — o que importa é o agregado.")
        st.download_button("📥 Baixar resultado da cesta (CSV)",
                           pd.concat([B["por_ativo"].set_index("Ativo"), B["estado"].set_index("Ativo")], axis=1)
                           .to_csv().encode("utf-8"), "cesta.csv", "text/csv")

# ---------------------------------------------------------------------
# ABA 4 — GUIA
# ---------------------------------------------------------------------
with tab4:
    st.title("📖 Guia da ferramenta")
    st.markdown("""
Este app faz três coisas: **mede** o estado técnico de criptomoedas com um score de 0 a 100, **simula** o que
teria acontecido se você seguisse esse score no passado, e **testa** se esse resultado é confiável ou sorte.
Nada aqui é recomendação de investimento — é um instrumento de medição, com as limitações descritas no final.
""")

    g1, g2, g3, g4, g5, g6, g7 = st.tabs(["1. O score", "2. Radar", "3. Backtesting", "4. Os testes de confiança",
                                          "5. Filtros e parâmetros", "6. O que já foi validado", "7. Cesta"])

    with g7:
        st.markdown("""
### Por que uma cesta, e como usar a aba

Os testes em 8 ativos mostraram que o motor **funciona onde há tendências longas** (SOL, DOGE, AVAX) e **não funciona
onde não há** (ADA, LINK, DOT) — e não existe forma de saber de antemão qual altcoin vai tender nos próximos anos.
A resposta não é escolher melhor: é **não escolher**. Numa cesta de peso igual, os acertos grandes de dois ou três
ativos pagam as perdas pequenas dos demais.

| 5 anos, 8 ativos, peso igual | Robô | Hold |
|---|---|---|
| Retorno | +111% | −55% |
| Drawdown máximo | −50% | −85% |
| Sharpe | 0,61 | 0,13 |
| **Fora da amostra** (dia 252 em diante) | **+127%, Sharpe 0,72** | +46%, Sharpe 0,47 |

Até a cesta só com os cinco ativos que individualmente reprovaram (ADA, DOGE, LINK, AVAX, DOT) rendeu +56% fora
da amostra contra −21% do hold.

**Como ler a aba:**
- **Hoje** — quantos ativos estão comprados (exposição da cesta) e o estado de cada um. Score acima do gatilho com
  posição "Em caixa" = compra vale a partir do próximo fechamento.
- **Desempenho** — curva da cesta robô vs. cesta hold; a linha tracejada marca onde começa o trecho fora da amostra.
- **Por ano** — a cesta robô perde para o hold nos anos de alta explosiva (custo de entrar depois do fundo) e vence
  com folga nos anos de queda. Se você não aceita ficar atrás em anos como 2023, seguidor de tendência não é para você.
- **Contribuição** — quanto cada ativo somou ao retorno da cesta. É normal 2–3 ativos concentrarem quase tudo.

**Regras práticas:** peso igual, rebalanceamento pelo menos semanal, não remover um ativo só porque ficou atrás
do hold num ano (é o comportamento esperado de parte da cesta), e não adicionar ativos com menos de 2 anos de
histórico — o robô precisa de ciclo completo para ser avaliado.
""")

    with g1:
        st.markdown("""
### Como o score é construído

O **Score Técnico** é uma média ponderada de quatro componentes, cada um de 0 a 100. Os pesos padrão
(RSI 10% · Tendência 35% · Momentum 20% · Risco 35%) foram os que passaram nos testes de confiança em 5 anos de SOL.

| Componente | Pergunta que responde | Cálculo | Leitura |
|---|---|---|---|
| **RSI (contrarian)** | O ativo caiu demais recentemente? | RSI de 14 dias invertido: RSI 30 → 100 pontos; RSI 70 → 0 pontos. | Pontua o **sobrevendido**. Peso pequeno: sozinho ele atrapalha (compra "faca caindo"), mas dentro do score segura a euforia. |
| **Tendência** | O preço está em alta estrutural? | Três checagens somadas: EMA 9 > EMA 21 (**35 pts**) · MACD > linha de sinal (**35 pts**) · preço > média de 50 dias (**30 pts**). | 100 = as três confirmam. É o componente com maior poder preditivo nos testes. |
| **Momentum** | Quanto subiu nos últimos 10 dias? | −10% → 0 · 0% → 50 · +10% → 100. **Acima de +15% o valor decai** (a +25% volta a 50, a +35% chega a 0). | Alta forte é boa até certo ponto; alta rápida demais historicamente precede queda. |
| **Risco** | O ativo está calmo ou nervoso? | 100 − (desvio-padrão dos retornos diários de 20 dias × 10). Vol diária de 3% → 70 pts; 8% → 20 pts. | Prefere entrar em mercados menos voláteis — reduz falsos rompimentos. |

**Score Final** (só no Radar) = 70% Score Técnico + 15% Sentimento das notícias + 15% Fear & Greed *invertido*
(medo extremo no mercado soma pontos — lógica contrária). O backtesting usa **só o Score Técnico**, porque não há
histórico confiável de sentimento para simular.

### Escala de decisão
| Score | Decisão | Score | Decisão |
|---|---|---|---|
| ≥ 80 | 🟢 Compra Forte | 41–44 | 🟠 Venda Fraca |
| 60–79 | 🟢 Compra Média | 21–40 | 🔴 Venda Média |
| 51–59 | 🟡 Compra Fraca | ≤ 20 | 🔴 Venda Forte |
| 45–50 | ⚪ Neutro | | |

O **robô do backtest** usa uma regra mais simples e com histerese: **compra quando o score sobe a 65 ou mais e vende
quando cai a 45 ou menos**; entre 45 e 65 mantém o que estava fazendo. Isso evita entrar e sair a cada oscilação.

**⚠️ Esticado** aparece quando o momentum de 10 dias passa de +15% ou o RSI de 75. É um aviso: nos dados históricos,
o retorno médio dos 20 dias seguintes a um score acima de 75 foi menor que o da faixa 55–75.
""")

    with g2:
        st.markdown("""
### Radar de Mercado

Duas tabelas com o **mesmo motor** em bases de tempo diferentes:

- **⭐ Watchlist — análise diária.** Usa 1 ano de fechamentos diários (Yahoo Finance). É a leitura oficial: é
  exatamente o que o backtesting simula. Escolha os ativos na sidebar e clique em *Salvar watchlist*.
- **🌐 Top 100 — score rápido.** Usa os 168 pontos horários dos últimos 7 dias (CoinGecko). Serve para varrer o mercado
  e achar candidatos, não para decidir: RSI e EMAs em horas se comportam diferente de dias.

**Cabeçalho:**
- **Fear & Greed** — índice público (alternative.me) de 0 (medo extremo) a 100 (ganância extrema).
- **Sentimento geral** — score 0–100 das manchetes do Cointelegraph. Com chave do Gemini na sidebar, a IA lê as
  manchetes e resume; sem chave, um dicionário de palavras positivas/negativas em português e inglês faz o trabalho.
- **Regime BTC** — aparece se o filtro estiver ligado: 🟢 BTC acima da sua média de 200 dias, 🔴 abaixo.

**Gráfico** — preços da watchlist; "Normalizar (base 100)" põe todos partindo de 100 para comparar desempenho.
**Detalhe por ativo** — evolução do score e do RSI, e as notícias que alimentaram o sentimento daquele ativo.
""")
        st.markdown(HELP_RADAR)

    with g3:
        st.markdown("""
### Backtesting — o que a simulação faz

1. Baixa o histórico diário do ativo (1y, 2y, 5y…).
2. Calcula o Score Técnico dia a dia, **olhando só para trás** (nenhum indicador usa dados futuros).
3. Aplica a regra: score ≥ gatilho de compra → fica comprado; score ≤ gatilho de venda → fica em caixa.
4. O sinal do fechamento de hoje só vale **a partir de amanhã** (não dá para comprar ao preço de fechamento que
   gerou o sinal).
5. Cobra a taxa configurada em cada entrada e cada saída.
6. Compara com **Buy & Hold**: comprar no primeiro dia e não fazer nada.

**Botões:**
- **▶️ Rodar simulação** — curva de retorno, métricas, lista de trades e CSV auditável (cada dia com todos os
  indicadores, o score, a posição e o retorno).
- **🧪 Walk-forward simples** — testa todas as combinações de gatilhos numa parte do histórico (treino) e mostra como
  cada uma se saiu na parte restante (teste). Se os melhores no treino não repetem no teste, os gatilhos estão
  ajustados ao passado.
- **🔁 Walk-forward rolante** — versão em janelas móveis (treina 1 ano, testa 1 trimestre, avança). Mostra também o
  gatilho fixo no mesmo trecho. Nos testes, o **gatilho fixo venceu a re-otimização** em todos os ativos: use isto
  para conferir robustez, não para escolher gatilhos.
- **🩺 Diagnóstico do sinal** — o score de hoje prevê o retorno dos próximos N dias? Correlação por componente e
  retorno médio por faixa de score.
- **🎲 Teste de permutação** — embaralha os retornos 200 vezes e roda o robô em cada série. Se o resultado real
  não é melhor que 95% dos embaralhados, foi sorte.
- **🏁 Veredito por ativo** — roda tudo isso para vários ativos de uma vez e classifica cada um.
""")
        st.markdown(HELP_BACKTEST)

    with g4:
        st.markdown("""
### Por que os testes de confiança existem

Um backtest com retorno de +300% não prova nada: com poucos trades, basta um acerto grande. Três perguntas
separam um resultado real de uma coincidência, e cada uma tem um botão:

**1. "Isso pode ter sido sorte?" → Teste de permutação.**
Embaralhamos a ordem dos retornos diários (a distribuição fica igual, mas a sequência — a tendência — some) e
rodamos o robô 200 vezes. Se o Sharpe real for maior que 95% dos Sharpes embaralhados, o **p-valor** é < 0,05 e há
evidência de que o robô captura algo real na *ordem* dos preços. Em SOL 5 anos: p = 0,007 ✅. Em BTC e ETH: p ≈ 0,17 ❌.

**2. "Funciona em dados que não foram usados para ajustar nada?" → Fora da amostra (OOS).**
Qualquer número calculado no mesmo período em que os parâmetros foram escolhidos é otimista. O trecho OOS
(a partir do dia 252, ou as janelas do rolante) é a única estimativa honesta do que esperar. Regra: **se o Sharpe
OOS não supera o do hold, o robô não compensa** naquele ativo.

**3. "O score antecipa o retorno ou só o descreve?" → Diagnóstico.**
Correlação entre o score de hoje e o retorno de 5/10/20 dias à frente. Em finanças, 0,05 já é sinal e 0,10 é forte.
O motor mostra sinal apenas em horizontes de 3–4 semanas; em 5 dias é ruído — por isso trades curtos tendem a perder.

**Como ler o veredito:**
- ✅ **Sinal real** — p < 0,05 **e** bate o hold fora da amostra. Usar o robô.
- 🛡️ **Só reduz drawdown** — sem evidência estatística, mas o drawdown OOS é menor e o Sharpe não é pior que o hold.
  O robô funciona como freio, não como acelerador.
- ❌ **Sem vantagem** — hold.

**Aviso sobre n pequeno:** menos de ~8 trades ou menos de 2 anos de dados não permitem concluir nada; o app avisa.
""")

    with g5:
        st.markdown("""
### Sidebar — filtros e parâmetros

**🔧 Parâmetros do motor** — períodos dos indicadores e pesos do score. Os padrões são os validados; mexer neles
exige rodar de novo permutação e OOS. Os pesos são normalizados automaticamente (só a proporção importa).

**🛡️ Filtros de risco**
- **Filtro de regime BTC** — só permite posição comprada quando o BTC está acima da sua média de 200 dias. Lógica:
  altcoins raramente sobem com BTC em baixa. Resultado nos testes: **reduz drawdown** (SOL 5y: −54% → −38%) mas
  **custa retorno fora da amostra** (+226% → +104%), porque SOL sai do fundo antes do BTC. Médias mais curtas
  (100/150) pioram. Desligado por padrão; ligue se prioriza risco menor.
- **Não comprar esticado** — bloqueia novas compras com o aviso ⚠️ ativo. **Piorou tudo** nos testes: bloqueia
  exatamente os rompimentos que pagam a estratégia. Desligado por padrão; existe para você comprovar.

**💾 Salvar parâmetros** — grava tudo em `parametros.json`; sem isso, recarregar o app volta aos padrões.
**↩️ Restaurar padrões validados** — apaga o arquivo.

**Gatilhos (aba Backtesting)** — 65/45 é um platô, não um pico: qualquer valor entre 65–75 / 45–55 dá resultado
parecido, e re-otimizar a cada trimestre piora. Trate como constante.

**Taxa por operação** — 0,10% cobre corretagem típica de exchange + um pouco de slippage. Cobrada na entrada e na saída.

### Sentimento histórico de notícias (5º componente, experimental)

O Radar usa o sentimento das notícias **de hoje** como informação. Para saber se sentimento **prevê** alguma coisa, é
preciso um histórico datado — e é isso que o script `build_sentiment_history.py` constrói:

1. Para cada ativo e cada semana dos últimos 5 anos, busca manchetes datadas no GDELT (base pública de notícias).
2. Pontua a semana de 0 a 100 com o léxico e, opcionalmente (`--ai`), com o Gemini.
3. Salva em `sentimento_historico.csv`, retomável (pode rodar em várias sessões).

Com o CSV na pasta do app, o backtest ganha o botão **🗞️ Testar sentimento**: roda o mesmo ativo sem e com o
componente (peso 0,25), compara retorno, Sharpe, fora da amostra e p-valor, e mostra a correlação do sentimento com o
retorno de 20 dias. **Sem look-ahead:** o score de uma semana só vale a partir da segunda-feira seguinte.

**Resultado em SOL (set/2021–mar/2026, tom diário do GDELT):**

| Uso do sentimento | Retorno | DD | Sharpe | Trades | Sharpe OOS | p |
|---|---|---|---|---|---|---|
| nenhum | +583% | −55% | 0,97 | 34 | 1,05 | 0,00 |
| como componente do score (0,25) | +229% | −72% | 0,72 | 33 | 0,78 | 0,08 |
| **como filtro de entrada (≥ 50)** | **+685%** | −55% | **1,15** | 21 | **1,18** | 0,00 |

- **No score, piora** em qualquer peso: o tom das notícias fica positivo com atraso e **segura o robô dentro das quedas**
  — atrapalha a saída, que é o que sustenta a estratégia. Peso padrão continua **0**.
- **Como confirmação de entrada, ajuda**: exige sentimento ≥ 50 para abrir compra (a venda não muda). Corta as
  entradas quase pela metade (rompimentos falsos) e sobe o Sharpe fora da amostra de 1,05 para 1,18. Validado em
  **um** ativo — por isso fica desligado por padrão, em "🗞️ Sentimento de notícias" na sidebar.
- O tom isolado tem correlação de **+0,15** com o retorno de 10–20 dias (a maior de todos os componentes), mas
  correlação alta não vira regra de trade automaticamente — é a lição da tarde inteira.
- **Volume de cobertura (atenção)** tem correlação **negativa** com o retorno futuro (−0,15), mas usá-lo para
  bloquear entradas destrói a estratégia: os rompimentos que pagam vêm com atenção alta. Descartado como filtro.

```
python build_sentiment_timeline.py --ativos SOL,BTC,ETH,DOGE,AVAX,ADA,LINK,DOT --anos 5   # recomendado: ~10 chamadas/ativo
python build_sentiment_history.py  --ativos SOL --anos 5 --ai   # semanal por manchetes + Gemini (lento: 1 chamada/semana)
```

**Dois coletores:** o `timeline` usa o tom que o próprio GDELT calcula sobre o texto completo dos artigos, dia a dia,
com uma chamada por semestre — rápido e sem esbarrar no limite de taxa. O `history` baixa manchetes semana a semana
e pontua com léxico/IA — mais controlável, mas o GDELT limita a ~1 chamada por minuto e o léxico converge para
neutro com muitas manchetes. Comece pelo `timeline`. No formato diário, o app suaviza o tom em 7 dias, normaliza pela
história do ativo (z-score móvel de 1 ano, só com o passado) e aplica com 1 dia de atraso.
""")

    with g6:
        st.markdown("""
### O que já foi validado (e o que foi derrubado)

Resultados com a configuração padrão, 5 anos (out/2021 – set/2026), taxa 0,1%, gatilhos 65/45, sem filtros:

| Ativo | Vol. anual | Robô / Hold | Drawdown robô / hold | p-valor | OOS robô / hold (Sharpe) | Veredito |
|---|---|---|---|---|---|---|
| **SOL** | 93% | +778% / −35% | −54% / −96% | **0,007** | 1,05 / 0,72 | ✅ Sinal real |
| **ETH** | 69% | +67% / −23% | −50% / −79% | 0,18 | 0,59 / 0,61 | 🛡️ Só reduz drawdown |
| **BTC** | 51% | +147% / +69% | −65% / −77% | 0,17 | 0,84 / 0,91 | ❌ Sem vantagem |
| **AVAX** | 94% | +183% / −83% | −62% / −96% | 0,05 | 0,53 / 0,24 | ✅ Sinal real |
| **DOGE** | 90% | +255% / −63% | −67% / −85% | 0,04 | 0,91 / 0,48 | ✅ Sinal real |
| **ADA** | 88% | −67% / −89% | −78% / −94% | 0,63 | −0,15 / 0,19 | ❌ Sem vantagem |
| **LINK** | 86% | −16% / −53% | −71% / −85% | 0,41 | 0,21 / 0,58 | ❌ Sem vantagem |
| **DOT** | 83% | −30% / −97% | −73% / −99% | 0,17 | 0,12 / −0,20 | 🛡️ Só reduz drawdown |

**Padrão observado (8 ativos):** volatilidade alta é *necessária* (os três ✅ são os três mais voláteis) mas *não
suficiente* (ADA, LINK e DOT são voláteis e reprovam). O que separa é ter tido **tendências de meses** no período —
e isso não é previsível. Três aprovações em oito com p < 0,05 é muito acima do acaso (esperado: 0,4), então o
sinal é real em geral; só não se sabe *onde* vai aparecer. Daí a aba 🧺 Cesta. Em **todos** os 8 ativos o robô
teve drawdown menor que o hold.

**Perfil dos trades (SOL):** win rate ~37%, ganho médio +41%, perda média −10%, payoff 4:1. Três ou quatro
tendências por ciclo pagam dezenas de perdas pequenas. **Isso exige aceitar sequências de 4–5 perdas seguidas** — quem
não aceita, não deve usar seguidor de tendência.

**Ideias testadas e derrubadas pelos dados** (todas pareciam boas):
- Zerar o peso do RSI (sugerido pela correlação) → p-valor foi de 0,007 para 0,42. O RSI segura a euforia dentro do score.
- Bloquear compras esticadas → piorou in-sample e OOS.
- Regime BTC com média mais curta → piorou OOS.
- Re-otimizar gatilhos a cada trimestre → perdeu para o fixo nos três ativos.

### Limitações honestas
- Cinco anos = um ciclo e meio de cripto. O futuro pode ter outro regime.
- Só posição comprada ou caixa; sem short, sem alavancagem, sem stop-loss intradiário.
- Preços de fechamento diário; a execução real terá slippage maior em altcoins pequenas.
- Sentimento e Fear & Greed não entram no backtest — o Score Final do Radar não está validado, só o Score Técnico.
- Nada aqui é recomendação de investimento.
""")
