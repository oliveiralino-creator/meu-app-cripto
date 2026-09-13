import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
import feedparser
import re

st.set_page_config(page_title="Crypto Market Intelligence", layout="wide", initial_sidebar_state="expanded")

# --- 1. MÓDULO DE NOTÍCIAS E SENTIMENTO ---

@st.cache_data(ttl=600)
def get_crypto_news():
    feed_url = "https://cointelegraph.com/rss"
    parsed_feed = feedparser.parse(feed_url)
    
    news_list = []
    for entry in parsed_feed.entries[:15]:
        news_list.append({
            'title': entry.title,
            'link': entry.link,
            'published': entry.published
        })
    return news_list

def analyze_sentiment(news_list):
    if not news_list: return 50
        
    positive_words = ['surge', 'rally', 'bull', 'bullish', 'jump', 'gain', 'adoption', 'approve', 'etf', 'high', 'buy', 'growth', 'upgrade']
    negative_words = ['plunge', 'crash', 'bear', 'bearish', 'drop', 'fall', 'hack', 'ban', 'sec', 'sue', 'sell', 'scam', 'delay', 'fear']
    
    sentiment_score = 50
    for news in news_list:
        title_lower = news['title'].lower()
        pos_count = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', title_lower))
        neg_count = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', title_lower))
        sentiment_score += (pos_count * 2) - (neg_count * 2)
        
    return max(0, min(100, sentiment_score))

latest_news = get_crypto_news()
market_sentiment = analyze_sentiment(latest_news)

# --- 2. FUNÇÕES TÉCNICAS ---

@st.cache_data(ttl=300)
def get_crypto_data():
    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
        "&sparkline=true&price_change_percentage=1h,24h,7d,30d"
    )
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200: return response.json()
    return []

def calculate_rsi(prices, period=14):
    if len(prices) < period: return 50
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        upval = delta if delta > 0 else 0.
        downval = -delta if delta < 0 else 0.
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi[-1]

def calculate_composite_score(row, market_sentiment_score):
    prices = row.get('sparkline_in_7d', {}).get('price', [])
    if not prices or len(prices) < 50: return 50, 50

    current_rsi = calculate_rsi(prices)
    if current_rsi <= 30: rsi_score = 100
    elif current_rsi >= 70: rsi_score = 0
    else: rsi_score = 100 - ((current_rsi - 30) * (100 / 40))

    sma_50 = np.mean(prices[-50:])
    ratio = prices[-1] / sma_50
    if ratio >= 1.05: trend_score = 100
    elif ratio <= 0.95: trend_score = 0
    else: trend_score = (ratio - 0.95) * 1000

    var_24h = row.get('price_change_percentage_24h_in_currency', 0)
    if pd.isna(var_24h): var_24h = 0
    if var_24h >= 5: mom_score = 100
    elif var_24h <= -5: mom_score = 0
    else: mom_score = (var_24h + 5) * 10

    tech_score = (0.40 * rsi_score) + (0.30 * trend_score) + (0.30 * mom_score)
    final_composite_score = (tech_score * 0.70) + (market_sentiment_score * 0.30)
    
    return max(0, min(100, final_composite_score)), tech_score

def classify_action(score):
    if score >= 80: return "🟢 Compra Forte"
    elif score >= 60: return "🟢 Compra Média"
    elif score >= 51: return "🟡 Compra Fraca"
    elif score >= 45: return "⚪ Neutro"
    elif score >= 41: return "🟠 Venda Fraca"
    elif score >= 21: return "🔴 Venda Média"
    else: return "🔴 Venda Forte"

# --- 3. PROCESSAMENTO DE DADOS ---

raw_data = get_crypto_data()

if raw_data:
    df = pd.DataFrame(raw_data)
    df['Ativo'] = df['symbol'].str.upper()
    
    scores = df.apply(lambda row: calculate_composite_score(row, market_sentiment), axis=1)
    df['Score Final (Composto)'] = [s[0] for s in scores]
    df['Score Apenas Técnico'] = [s[1] for s in scores]
    df['Ação Sugerida'] = df['Score Final (Composto)'].apply(classify_action)
    df['RSI 14h'] = df.apply(lambda r: calculate_rsi(r.get('sparkline_in_7d', {}).get('price', [])), axis=1).round(1)

    # --- 4. INTERFACE ---
    st.sidebar.title("Filtros")
    ativos_selecionados = st.sidebar.multiselect("Filtrar Moedas (Exibe o Gráfico):", options=df['Ativo'].tolist(), default=[])
    
    st.title("📊 Crypto Market Intelligence")
    
    colA, colB = st.columns([1, 3])
    colA.metric(
        "Sentimento Geral (Notícias)", 
        f"{market_sentiment:.0f}/100", 
        "Otimista" if market_sentiment >= 55 else "Pessimista" if market_sentiment <= 45 else "Neutro",
        delta_color="normal" if market_sentiment >= 50 else "inverse"
    )
    colB.info("O **Score Final** combina **70%** da Análise Técnica do ativo e **30%** do Sentimento das notícias das últimas horas.")
    
    st.divider()

    df_view = df[df['Ativo'].isin(ativos_selecionados)] if ativos_selecionados else df.copy()

    # --- GRÁFICO REINTEGRADO ---
    if ativos_selecionados:
        st.subheader("📈 Evolução nos Últimos 7 Dias")
        chart_data = []
        for index, row in df_view.iterrows():
            prices = row.get('sparkline_in_7d', {}).get('price', [])
            if prices:
                for i, price in enumerate(prices):
                    chart_data.append({'Ativo': row['Ativo'], 'Hora': i, 'Preço USD': price})
        
        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            fig_line = px.line(df_chart, x='Hora', y='Preço USD', color='Ativo', template='plotly_white')
            fig_line.update_xaxes(showticklabels=False, title="Linha do tempo (168 horas)") 
            st.plotly_chart(fig_line, use_container_width=True)

    # --- TABELA ---
    cols_to_keep = {
        'Ativo': 'Ativo',
        'current_price': 'Preço (USD)',
        'price_change_percentage_24h_in_currency': 'Var 24h (%)',
        'RSI 14h': 'RSI 14h',
        'Score Apenas Técnico': 'Score Técnico',
        'Score Final (Composto)': 'Score Final',
        'Ação Sugerida': 'Ação Sugerida'
    }
    df_clean = df_view.rename(columns=cols_to_keep)[list(cols_to_keep.values())]

    st.subheader("Painel de Decisão Consolidado")
    st.dataframe(
        df_clean.style.format({
            "Preço (USD)": "${:,.4f}",
            "Var 24h (%)": "{:.2f}%",
            "RSI 14h": "{:.1f}",
            "Score Técnico": "{:.0f}",
            "Score Final": "{:.0f}"
        }).background_gradient(subset=['Score Final'], cmap='RdYlGn', vmin=0, vmax=100),
        use_container_width=True,
        hide_index=True
    )

    # --- 5. RODAPÉ DE NOTÍCIAS ---
    st.divider()
    st.subheader("📰 Radar de Notícias Recentes (Feed CoinTelegraph)")
    for news in latest_news[:5]:
        st.markdown(f"- [{news['title']}]({news['link']}) *(Publicado em: {news['published']})*")

else:
    st.error("Falha ao conectar com a API do CoinGecko.")