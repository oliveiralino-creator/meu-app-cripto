# Crypto Market Intelligence

App Streamlit que **mede** o estado técnico de criptomoedas com um score 0–100, **simula** o que teria acontecido
seguindo esse score no passado e **testa** se o resultado é confiável ou sorte.

> Instrumento de medição, não recomendação de investimento. Leia a aba **📖 Guia** dentro do app antes de usar.

## Rodar

```bash
pip install -r requirements.txt
streamlit run app_cripto.py
```

Opcional: chave da API do Gemini na sidebar para o sentimento de notícias com IA (sem chave, usa um léxico pt/en).

## Estrutura

| Aba | O que faz |
|---|---|
| 📊 Radar de Mercado | Score diário da watchlist (mesmo motor do backtest) + varredura horária do Top 100 |
| 💼 Simulador de Carteira | Registro de posições virtuais, fechamento, P&L realizado e não realizado |
| ⏪ Backtesting | Simulação com taxas e métricas de risco, walk-forward, diagnóstico do sinal, teste de permutação e veredito por ativo |
| 🧺 Cesta | Robô aplicado a vários ativos com peso igual: exposição de hoje, equity vs hold, contribuição por ativo |
| 📖 Guia | Explicação de cada variável, métrica, teste e filtro; resultados validados e limitações |

## O score em uma linha

`Score = 0,10·RSI_contrarian + 0,35·Tendência + 0,20·Momentum + 0,35·Risco` (cada componente 0–100).
Robô: compra com score ≥ 65, vende com ≤ 45. Detalhes na aba Guia.

## Resultados validados (5 anos, gatilhos 65/45, taxa 0,1%)

| Ativo | Robô / Hold | Drawdown robô / hold | p-valor (permutação) | Veredito |
|---|---|---|---|---|
| SOL | +778% / −35% | −54% / −96% | 0,007 | ✅ Sinal real |
| ETH | +67% / −23% | −50% / −79% | 0,18 | 🛡️ Só reduz drawdown |
| BTC | +147% / +69% | −65% / −77% | 0,17 | ❌ Sem vantagem |
| AVAX | +183% / −83% | −62% / −96% | 0,05 | ✅ Sinal real |
| DOGE | +255% / −63% | −67% / −85% | 0,04 | ✅ Sinal real |
| ADA | −67% / −89% | −78% / −94% | 0,63 | ❌ Sem vantagem |
| LINK | −16% / −53% | −71% / −85% | 0,41 | ❌ Sem vantagem |
| DOT | −30% / −97% | −73% / −99% | 0,17 | 🛡️ Só reduz drawdown |

**Cesta dos 8 (peso igual):** robô +111% vs hold −55%; drawdown −50% vs −85%; fora da amostra Sharpe 0,72 vs 0,47.
O motor funciona onde há tendências longas (não previsível de antemão) — por isso o uso recomendado é em cesta.
Em todos os 8 ativos o robô teve drawdown menor que o hold. Gatilhos são constantes:
re-otimizar perdeu para o fixo em todos os ativos. Filtros de regime BTC e "não comprar esticado" existem, mas
ficam desligados por padrão porque pioraram o resultado fora da amostra.

## Sentimento histórico (experimental)

`build_sentiment_history.py` coleta manchetes datadas (GDELT, gratuito) por ativo e semana, pontua (léxico e/ou
Gemini com `--ai`) e gera `sentimento_historico.csv`. Com o CSV na pasta, o backtest testa o sentimento como quinto
componente do score ou como filtro de entrada (botão **🗞️ Testar sentimento**), sem look-ahead.

Resultado em SOL: como componente do score **piora** (atrasa saídas; peso fica 0); como **filtro de entrada**
(sentimento ≥ 50) melhora — Sharpe fora da amostra 1,05 → 1,18 com metade das entradas. Opção na sidebar, desligada
por padrão até validar em mais ativos.

```bash
python build_sentiment_timeline.py --ativos SOL,BTC,ETH,DOGE,AVAX,ADA,LINK,DOT --anos 5   # recomendado (tom diário, poucas chamadas)
python build_sentiment_history.py  --ativos SOL --anos 5 --ai                            # alternativa semanal por manchetes + Gemini
```

## Arquivos gerados

- `parametros.json` — parâmetros salvos pela sidebar (opcional)
- `watchlist.json` — watchlist salva
- `carteira_virtual.csv` — posições do simulador
- `sentimento_historico.csv` — sentimento semanal por ativo (gerado pelo script acima)

Em hospedagem efêmera (Streamlit Cloud) esses arquivos se perdem ao reiniciar; use os botões de download.
