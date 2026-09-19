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

O motor funciona em ativos de beta alto; em BTC/ETH apenas reduz drawdown. Gatilhos são constantes:
re-otimizar perdeu para o fixo em todos os ativos. Filtros de regime BTC e "não comprar esticado" existem, mas
ficam desligados por padrão porque pioraram o resultado fora da amostra.

## Arquivos gerados

- `parametros.json` — parâmetros salvos pela sidebar (opcional)
- `watchlist.json` — watchlist salva
- `carteira_virtual.csv` — posições do simulador

Em hospedagem efêmera (Streamlit Cloud) esses arquivos se perdem ao reiniciar; use os botões de download.
