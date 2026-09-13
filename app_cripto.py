import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import feedparser
import re
import urllib.parse
import google.generativeai as genai
from datetime import datetime, timedelta, timezone
import os
import yfinance as yf

st.set_page_config(page_title="Crypto Market Intelligence", layout="wide", initial_sidebar_state="expanded")

# --- 1. APIS MACRO E DE DADOS ---
@st.cache_data(ttl=300)
def get_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=true&price_change_percentage=1h,24h,7d,30d"
    try:
        response = requests.get(url, headers={"accept": "application/json"})
        if response.status_code == 200: return response.json()
    except: pass
    return []

@st.cache_data(ttl=3600)
def get_fear_and_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1")
        if r.status_code == 200:
            return int(r.json()['data'][0]['value']), r.json()['data'][0]['value_classification']
    except: pass
    return 50, "Neutral"

@st.cache_data(ttl=600)
def get_crypto_news(ativos=[]):
    news_list = []
    try:
        if not ativos: feed_url = "https://cointelegraph.com/rss"
        else:
            query = " OR ".join([f"{coin} criptomoeda" for coin in ativos[:3]])
            feed_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
            
        parsed_feed = feedparser.parse(feed_url)
        limite_dias = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=3)
        
        for entry in parsed_feed.entries:
            try:
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    dt_pub = datetime(*entry.published_parsed[:6])
                    if dt_pub < limite_dias: continue
                    dt_brasil = dt_pub - timedelta(hours=3)
                    data_formatada = dt_brasil.strftime("%d/%m/%Y %H:%M")
                else: data_formatada = "Recente"
            except: data_formatada = "Recente"

            news_list.append({'title': entry.title, 'link': entry.link, 'published': data_formatada})
            if len(news_list) >= 15: break
    except: pass
    return news_list

# --- 2. MOTORES DE SENTIMENTO E IA ---
def analyze_sentiment_lexical(news_list):
    if not news_list: return 50
    pos_words = ['surge', 'rally', 'bull', 'bullish', 'jump', 'gain', 'adoption', 'etf', 'alta', 'crescimento', 'lucro', 'dispara', 'aprova']
    neg_words = ['plunge', 'crash', 'bear', 'bearish', 'drop', 'fall', 'hack', 'ban', 'sec', 'queda', 'tombo', 'roubo', 'cai', 'despenca']
    score = 50
    for news in news_list:
        title = news['title'].lower()
        score += (sum(1 for w in pos_words if re.search(r'\b'+w+r'\b', title)) * 2)
        score -= (sum(1 for w in neg_words if re.search(r'\b'+w+r'\b', title)) * 2)
    return max(0, min(100, score))

def analyze_sentiment_ai(news_list, api_key):
    if not news_list: return 50, "Sem notícias recentes suficientes para análise."
    try:
        genai.configure(api_key=api_key)
        valid_model_name = next((m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods), None)
        if not valid_model_name: return analyze_sentiment_lexical(news_list), "Erro: Nenhum modelo de IA disponível."
            
        model = genai.GenerativeModel(valid_model_name)
        titles = "\n".join([f"- {n['title']}" for n in news_list])
        prompt = f"Atue como um analista quantitativo. Avalie as manchetes:\n{titles}\nREGRA: Responda EXATAMENTE em 1 linha: NUMERO|RESUMO_EM_PORTUGUES"
        
        texto_ia = model.generate_content(prompt).text.strip()
        if '|' in texto_ia:
            parts = texto_ia.split('|', 1)
            num_str = re.sub(r'\D', '', parts[0])
            return min(100, max(0, int(num_str) if num_str else analyze_sentiment_lexical(news_list))), parts[1].strip()
        else: return analyze_sentiment_lexical(news_list), "IA falhou no formato. Usando algoritmo básico."
    except Exception as e: return analyze_sentiment_lexical(news_list), f"Erro na IA: {str(e)}"

# --- 3. MATEMÁTICA FINANCEIRA ---
def calculate_rsi(prices, period=14):
    if len(prices) < period: return 50
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up, down = seed[seed >= 0].sum() / period, -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        up, down = (up * (period - 1) + (delta if delta > 0 else 0.)) / period, (down * (period - 1) + (-delta if delta < 0 else 0.)) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi[-1]

def calculate_composite_score(row, sentiment_score, fg_score):
    prices = row.get('sparkline_in_7d', {}).get('price', [])
    if not prices or len(prices) < 50: return 50, 50

    current_rsi = calculate_rsi(prices)
    rsi_score = 100 if current_rsi <= 30 else (0 if current_rsi >= 70 else 100 - ((current_rsi - 30) * 2.5))
    
    ratio = prices[-1] / np.mean(prices[-50:])
    trend_score = 100 if ratio >= 1.05 else (0 if ratio <= 0.95 else (ratio - 0.95) * 1000)
    
    var_24h = row.get('price_change_percentage_24h_in_currency', 0)
    mom_score = 100 if var_24h >= 5 else (0 if var_24h <= -5 else (var_24h + 5) * 10)

    vol_score = min(100, (row.get('total_volume', 0) / max(row.get('market_cap', 1), 1)) * 500)
    tech_score = (0.35 * rsi_score) + (0.30 * trend_score) + (0.20 * mom_score) + (0.15 * vol_score)
    final_score = (tech_score * 0.60) + (sentiment_score * 0.20) + ((100 - fg_score) * 0.20)
    return max(0, min(100, final_score)), tech_score

def classify_action(score):
    if score >= 80: return "🟢 Compra Forte"
    elif score >= 60: return "🟢 Compra Média"
    elif score >= 51: return "🟡 Compra Fraca"
    elif score >= 45: return "⚪ Neutro"
    elif score >= 41: return "🟠 Venda Fraca"
    elif score >= 21: return "🔴 Venda Média"
    else: return "🔴 Venda Forte"

# --- EXECUÇÃO PRINCIPAL ---
raw_data = get_crypto_data()
df = pd.DataFrame(raw_data) if raw_data else pd.DataFrame()
if not df.empty:
    df['Ativo'] = df['symbol'].str.upper()
    lista_ativos = df['Ativo'].tolist()
else: lista_ativos = []

st.sidebar.title("Configurações e IA")
ia_key = st.sidebar.text_input("Gemini API Key:", type="password")
st.sidebar.divider()
ativos_selecionados = st.sidebar.multiselect("Filtrar Moedas no Radar:", options=lista_ativos, default=[])

# --- ABAS DO APLICATIVO ---
tab1, tab2, tab3 = st.tabs(["📊 Radar de Mercado", "💼 Simulador de Carteira", "⏪ Backtesting Técnico"])

# ==========================================
# ABA 1: RADAR DE MERCADO (Gráfico e Variação 1h Reintegrados)
# ==========================================
with tab1:
    latest_news = get_crypto_news(ativos_selecionados)
    fg_value, fg_class = get_fear_and_greed()

    if ia_key: noticia_score, ia_status = analyze_sentiment_ai(latest_news, ia_key)
    else: noticia_score, ia_status = analyze_sentiment_lexical(latest_news), "⚙️ IA Desativada."

    col1, col2, col3 = st.columns(3)
    col1.metric("Fear & Greed Index (Macro)", f"{fg_value}/100", fg_class)
    col2.metric("Sentimento (Filtro)" if ativos_selecionados else "Sentimento Geral", f"{noticia_score:.0f}/100", "Positivo" if noticia_score > 50 else "Negativo")
    col3.info(f"🧠 {ia_status}")
    st.divider()

    if not df.empty:
        scores = df.apply(lambda row: calculate_composite_score(row, noticia_score, fg_value), axis=1)
        df['Score Final'] = [s[0] for s in scores]
        df['Ação Sugerida'] = df['Score Final'].apply(classify_action)
        df['Vol/Cap (%)'] = (df['total_volume'] / df['market_cap'] * 100).round(2)
        
        df_view = df[df['Ativo'].isin(ativos_selecionados)] if ativos_selecionados else df.copy()

        # O Gráfico voltou! (Aparece se houverem moedas filtradas)
        if ativos_selecionados and len(ativos_selecionados) <= 10:
            st.subheader(f"📈 Evolução de Preços - {', '.join(ativos_selecionados)}")
            chart_data = []
            for index, row in df_view.iterrows():
                prices = row.get('sparkline_in_7d', {}).get('price', [])
                for i, price in enumerate(prices):
                    chart_data.append({'Ativo': row['Ativo'], 'Hora': i, 'Preço USD': price})
            if chart_data:
                df_chart = pd.DataFrame(chart_data)
                fig_line = px.line(df_chart, x='Hora', y='Preço USD', color='Ativo', template='plotly_white')
                fig_line.update_xaxes(showticklabels=False, title="Linha do Tempo (Últimos 7 dias)") 
                st.plotly_chart(fig_line, use_container_width=True)
                st.divider()

        # Configuração da nova tabela (Agora com 1h incluído)
        cols_to_keep = {
            'Ativo': 'Ativo', 
            'current_price': 'Preço (USD)', 
            'price_change_percentage_1h_in_currency': 'Var 1h (%)', # Novo dado adicionado
            'price_change_percentage_24h_in_currency': 'Var 24h (%)', 
            'Vol/Cap (%)': 'Vol/Cap (%)', 
            'Score Final': 'Score Final', 
            'Ação Sugerida': 'Decisão Mestre'
        }
        df_clean = df_view.rename(columns=cols_to_keep)[list(cols_to_keep.values())]

        st.dataframe(
            df_clean.style.format({
                "Preço (USD)": "${:,.4f}", 
                "Var 1h (%)": "{:.2f}%", 
                "Var 24h (%)": "{:.2f}%", 
                "Vol/Cap (%)": "{:.1f}%", 
                "Score Final": "{:.0f}"
            }).background_gradient(subset=['Score Final'], cmap='RdYlGn', vmin=0, vmax=100), 
            use_container_width=True, 
            hide_index=True
        )

# ==========================================
# ABA 2: SIMULADOR DE CARTEIRA
# ==========================================
with tab2:
    st.subheader("Simulador de Posições (Forward-Testing)")
    st.markdown("Registre entradas virtuais baseadas nos Scores para acompanhar a evolução ao longo dos dias.")
    
    ARQUIVO_CARTEIRA = 'carteira_virtual.csv'
    
    with st.form("form_ordem"):
        c1, c2, c3 = st.columns(3)
        ativo_sim = c1.selectbox("Ativo", options=lista_ativos if lista_ativos else ['BTC', 'ETH'])
        preco_atual_sugerido = df[df['Ativo'] == ativo_sim]['current_price'].values[0] if not df.empty and ativo_sim in lista_ativos else 0.0
        preco_compra = c2.number_input("Preço de Compra (USD)", min_value=0.0, value=float(preco_atual_sugerido), format="%.4f")
        quantidade = c3.number_input("Quantidade de Moedas", min_value=0.0, value=1.0, format="%.4f")
        submit_ordem = st.form_submit_button("🛒 Registrar Compra Virtual")
        
        if submit_ordem and quantidade > 0:
            nova_ordem = pd.DataFrame({'Data': [datetime.now().strftime("%Y-%m-%d %H:%M")], 'Ativo': [ativo_sim], 'Preço_Compra': [preco_compra], 'Quantidade': [quantidade]})
            if os.path.exists(ARQUIVO_CARTEIRA):
                df_cart = pd.read_csv(ARQUIVO_CARTEIRA)
                df_cart = pd.concat([df_cart, nova_ordem], ignore_index=True)
            else:
                df_cart = nova_ordem
            df_cart.to_csv(ARQUIVO_CARTEIRA, index=False)
            st.success(f"Posição de {quantidade} {ativo_sim} registrada com sucesso!")

    st.divider()
    if os.path.exists(ARQUIVO_CARTEIRA) and not df.empty:
        df_cart = pd.read_csv(ARQUIVO_CARTEIRA)
        precos_atuais = df[['Ativo', 'current_price']].rename(columns={'current_price': 'Preço_Atual'})
        df_cart = df_cart.merge(precos_atuais, on='Ativo', how='left')
        
        df_cart['Total_Investido'] = df_cart['Preço_Compra'] * df_cart['Quantidade']
        df_cart['Valor_Atual'] = df_cart['Preço_Atual'] * df_cart['Quantidade']
        df_cart['Lucro/Prejuizo (USD)'] = df_cart['Valor_Atual'] - df_cart['Total_Investido']
        df_cart['Retorno (%)'] = (df_cart['Lucro/Prejuizo (USD)'] / df_cart['Total_Investido']) * 100
        
        st.markdown(f"### Desempenho Global: **${df_cart['Lucro/Prejuizo (USD)'].sum():,.2f}**")
        st.dataframe(df_cart.style.format({
            "Preço_Compra": "${:,.4f}", "Preço_Atual": "${:,.4f}",
            "Total_Investido": "${:,.2f}", "Valor_Atual": "${:,.2f}",
            "Lucro/Prejuizo (USD)": "${:,.2f}", "Retorno (%)": "{:.2f}%"
        }).background_gradient(subset=['Retorno (%)'], cmap='RdYlGn', vmin=-10, vmax=10), use_container_width=True, hide_index=True)
        
        if st.button("🗑️ Limpar Carteira Virtual"):
            os.remove(ARQUIVO_CARTEIRA)
            st.rerun()

# ==========================================
# ABA 3: BACKTESTING TÉCNICO (PARAMETRIZADO)
# ==========================================
with tab3:
    st.subheader("Motor de Backtesting (Otimização de Parâmetros)")
    st.markdown("Calibre os níveis de exigência do robô para encontrar o ponto de equilíbrio.")
    
    colA, colB = st.columns(2)
    ativo_bt = colA.selectbox("Ativo para Testar:", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"])
    periodo_bt = colB.selectbox("Período Histórico:", ["1y", "6mo", "2y", "max"])
    
    st.markdown("#### ⚙️ Calibração do Algoritmo")
    col_slider1, col_slider2 = st.columns(2)
    gatilho_compra = col_slider1.slider("Gatilho de Compra (Quão forte deve ser a tendência?):", min_value=50, max_value=90, value=80, step=5)
    gatilho_venda = col_slider2.slider("Gatilho de Venda (Quão fraca deve ficar a tendência?):", min_value=30, max_value=60, value=60, step=5)
    
    if st.button("▶️ Rodar Simulação Parametrizada"):
        with st.spinner("Processando dados e aplicando matriz de decisão..."):
            hist = yf.download(ativo_bt, period=periodo_bt, progress=False)
            if not hist.empty:
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                
                hist['RSI'] = calculate_rsi(hist['Close'].values, period=14)
                hist['EMA_9'] = hist['Close'].ewm(span=9, adjust=False).mean()
                hist['EMA_21'] = hist['Close'].ewm(span=21, adjust=False).mean()
                exp1, exp2 = hist['Close'].ewm(span=12, adjust=False).mean(), hist['Close'].ewm(span=26, adjust=False).mean()
                hist['MACD'] = exp1 - exp2
                hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
                
                hist['Score_BT'] = 50
                hist.loc[hist['RSI'] < 40, 'Score_BT'] += 10 
                hist.loc[hist['RSI'] > 75, 'Score_BT'] -= 20
                hist.loc[hist['EMA_9'] > hist['EMA_21'], 'Score_BT'] += 20
                hist.loc[hist['EMA_9'] <= hist['EMA_21'], 'Score_BT'] -= 20
                hist.loc[hist['MACD'] > hist['Signal_Line'], 'Score_BT'] += 20
                hist.loc[hist['MACD'] <= hist['Signal_Line'], 'Score_BT'] -= 20
                
                hist['Sinal_Temporario'] = np.nan
                hist.loc[hist['Score_BT'] >= gatilho_compra, 'Sinal_Temporario'] = 1 
                hist.loc[hist['Score_BT'] <= gatilho_venda, 'Sinal_Temporario'] = 0 
                hist['Posicao'] = hist['Sinal_Temporario'].ffill().fillna(0)
                
                hist['Retorno_Ativo'] = hist['Close'].pct_change()
                hist['Retorno_Robo'] = hist['Retorno_Ativo'] * hist['Posicao'].shift(1)
                hist['Acumulado_Hold'] = (1 + hist['Retorno_Ativo']).cumprod() - 1
                hist['Acumulado_Robo'] = (1 + hist['Retorno_Robo']).cumprod() - 1
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Acumulado_Hold']*100, mode='lines', name='Buy & Hold (%)', line=dict(color='gray')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Acumulado_Robo']*100, mode='lines', name='Robô Parametrizado (%)', line=dict(color='green', width=2)))
                fig.update_layout(title=f"Performance com Compra >= {gatilho_compra} e Venda <= {gatilho_venda}", yaxis_title="Retorno (%)", template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
                lucro_bh, lucro_est = hist['Acumulado_Hold'].iloc[-1] * 100, hist['Acumulado_Robo'].iloc[-1] * 100
                mudancas_posicao = hist['Posicao'].diff().abs().sum() / 2
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Resultado Buy & Hold", f"{lucro_bh:.2f}%")
                c2.metric("Resultado Robô", f"{lucro_est:.2f}%", f"{lucro_est - lucro_bh:.2f}% de diferença")
                c3.metric("Quantidade de Trades", f"{int(mudancas_posicao)}")

                df_export = hist[['Close', 'RSI', 'EMA_9', 'EMA_21', 'MACD', 'Score_BT', 'Posicao', 'Retorno_Robo']].dropna().round(4)
                st.download_button(label="📥 Baixar Série Histórica Auditável", data=df_export.to_csv().encode('utf-8'), file_name=f'backtest_parametrizado_{ativo_bt}.csv', mime='text/csv')
            else: st.error("Falha ao puxar dados.")