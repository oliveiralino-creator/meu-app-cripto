"""
Constrói um histórico SEMANAL de sentimento de notícias por ativo, para o backtest testar se sentimento
tem poder preditivo.

Fonte de manchetes datadas: GDELT DOC 2.0 (gratuita, sem chave, cobre 2017+). Uma consulta por ativo por semana.
Pontuação: léxico (sempre) e, opcionalmente, Gemini (--ai) — score 0–100, 50 = neutro.

Uso:
    python build_sentiment_history.py --ativos SOL,BTC,ETH,DOGE,AVAX,ADA,LINK,DOT --anos 5
    python build_sentiment_history.py --ativos SOL --anos 5 --ai --modelo gemini-2.5-flash-lite

Saída: sentimento_historico.csv (uma linha por ativo × semana). O script é retomável: rode de novo e ele
continua de onde parou. Com --ai, semanas já pontuadas pela IA não são reenviadas.
Coloque o CSV na mesma pasta do app_cripto.py; o backtest passa a oferecer o componente "Sentimento".

Custo/tempo: ~1,3 s por consulta GDELT → 8 ativos × 260 semanas ≈ 45 min. Gemini: uma chamada por semana/ativo;
no plano gratuito, use --ai em sessões de ~200 semanas por dia ou o modelo -lite.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# --- termos de busca por ativo (nomes que aparecem em manchetes) ---
TERMOS = {
    "BTC": '"bitcoin"', "ETH": '"ethereum"', "SOL": '"solana"', "DOGE": '"dogecoin"', "AVAX": '"avalanche" crypto',
    "ADA": '"cardano"', "LINK": '"chainlink"', "DOT": '"polkadot"', "XRP": '"ripple" OR "XRP"', "BNB": '"binance coin" OR "BNB"',
    "MATIC": '"polygon" crypto', "NEAR": '"near protocol"', "SUI": '"sui" blockchain', "LTC": '"litecoin"',
    "TRX": '"tron" crypto', "ATOM": '"cosmos" crypto', "UNI": '"uniswap"', "AAVE": '"aave"', "ARB": '"arbitrum"', "OP": '"optimism" crypto',
}

# --- léxico (mesmo do app) ---
POS_WORDS = {"surge", "rally", "bull", "bullish", "jump", "gain", "gains", "adoption", "etf", "approve", "approves",
             "record", "upgrade", "partnership", "inflow", "inflows", "soars", "soar", "rises", "climbs", "breakout",
             "alta", "crescimento", "lucro", "dispara", "aprova", "aprovação", "recorde", "adoção", "valoriza",
             "sobe", "avança", "entrada", "parceria", "otimista"}
NEG_WORDS = {"plunge", "crash", "bear", "bearish", "drop", "fall", "falls", "hack", "hacked", "ban", "bans",
             "lawsuit", "sues", "exploit", "outflow", "outflows", "liquidation", "liquidations", "slumps", "tumbles",
             "sinks", "dumps", "scam", "fraud", "collapse", "bankruptcy",
             "queda", "tombo", "roubo", "cai", "despenca", "hacker", "golpe", "processo", "proíbe", "saída",
             "liquidação", "liquidações", "pessimista", "desvaloriza", "recua"}
POS_PHRASES = ["sec aprova", "sec approves", "aprova etf", "approves etf", "etf approval", "all-time high"]
NEG_PHRASES = ["sec processa", "sec sues", "sec rejeita", "sec rejects", "sec charges"]


def sentiment_lexical(titles):
    if not titles:
        return None
    total = 0.0
    for t in titles:
        t = t.lower()
        s = 2 * sum(p in t for p in POS_PHRASES) - 2 * sum(p in t for p in NEG_PHRASES)
        words = set(re.findall(r"[a-záéíóúãõâêôç\-]+", t))
        s += len(words & POS_WORDS) - len(words & NEG_WORDS)
        total += max(-3, min(3, s))
    return round(50 + (total / len(titles)) * (50 / 3), 1)


# --- GDELT ---
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"


def gdelt_headlines(query, start, end, maxrecords=75, retries=6):
    params = {"query": f"{query} sourcelang:eng", "mode": "artlist", "maxrecords": maxrecords, "sort": "hybridrel",
              "format": "json", "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S")}
    wait = 5
    for i in range(retries):
        try:
            r = requests.get(GDELT, params=params, timeout=30, headers={"User-Agent": "crypto-sentiment-research/1.0"})
            if r.status_code == 200:
                try:
                    arts = r.json().get("articles", [])
                except ValueError:
                    return []                      # GDELT devolve texto vazio quando não há resultados
                titles, seen = [], set()
                for a in arts:
                    t = (a.get("title") or "").strip()
                    key = t.lower()[:80]
                    if t and key not in seen:
                        seen.add(key); titles.append(t)
                return titles
            if r.status_code == 429:
                print(f"    429 (limite) — aguardando {wait}s", flush=True); time.sleep(wait); wait = min(wait * 2, 90); continue
            print(f"    HTTP {r.status_code}", flush=True); time.sleep(wait)
        except requests.RequestException as e:
            print(f"    rede: {e}", flush=True); time.sleep(wait)
    return None


# --- Gemini ---
def gemini_client(api_key):
    from google import genai
    return genai.Client(api_key=api_key)


def sentiment_ai(client, model, asset, titles, retries=4):
    from google.genai import types
    prompt = ("Você é um analista quantitativo de criptomoedas. Estas são manchetes de UMA semana sobre "
              f"{asset}. Dê um score de sentimento de mercado para o ativo de 0 (extremamente negativo) a 100 "
              "(extremamente positivo), 50 = neutro. Responda só o JSON.\n\n" + "\n".join(f"- {t}" for t in titles[:60]))
    schema = {"type": "OBJECT", "properties": {"score": {"type": "INTEGER"}}, "required": ["score"]}
    wait = 10
    for _ in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt,
                                                  config=types.GenerateContentConfig(response_mime_type="application/json",
                                                                                     response_schema=schema, temperature=0.1))
            return int(max(0, min(100, json.loads(resp.text)["score"])))
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                print(f"    Gemini limite — aguardando {wait}s", flush=True); time.sleep(wait); wait = min(wait * 2, 120)
            else:
                print(f"    Gemini erro: {msg[:120]}", flush=True); time.sleep(3)
    return None


# --- principal ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ativos", default="SOL,BTC,ETH,DOGE,AVAX,ADA,LINK,DOT")
    ap.add_argument("--anos", type=float, default=5)
    ap.add_argument("--saida", default="sentimento_historico.csv")
    ap.add_argument("--ai", action="store_true", help="pontuar também com Gemini (precisa GEMINI_API_KEY)")
    ap.add_argument("--modelo", default="gemini-2.5-flash-lite")
    ap.add_argument("--pausa", type=float, default=1.3, help="segundos entre consultas GDELT")
    ap.add_argument("--max-semanas", type=int, default=0, help="limite de semanas por execução (0 = todas)")
    args = ap.parse_args()

    ativos = [a.strip().upper() for a in args.ativos.split(",") if a.strip()]
    hoje = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    inicio = hoje - timedelta(days=int(args.anos * 365))
    inicio -= timedelta(days=inicio.weekday())          # alinha na segunda-feira
    semanas = pd.date_range(inicio, hoje - timedelta(days=7), freq="7D")

    cols = ["data", "ativo", "n_manchetes", "sent_lexico", "sent_ia", "fonte"]
    if os.path.exists(args.saida):
        df = pd.read_csv(args.saida, parse_dates=["data"])
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
    else:
        df = pd.DataFrame(columns=cols)
    feitos = set(zip(df["ativo"], pd.to_datetime(df["data"]).dt.strftime("%Y-%m-%d")))

    client = None
    if args.ai:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("Defina GEMINI_API_KEY no ambiente para usar --ai")
        client = gemini_client(key)

    novas, n_exec = [], 0
    print(f"{len(ativos)} ativos × {len(semanas)} semanas | já feitos: {len(feitos)} | saída: {args.saida}")
    try:
        for a in ativos:
            termo = TERMOS.get(a, f'"{a}" crypto')
            for ws in semanas:
                chave = (a, ws.strftime("%Y-%m-%d"))
                if chave in feitos:
                    # com --ai, completa linhas que ainda não têm sent_ia
                    if args.ai:
                        m = (df["ativo"] == a) & (pd.to_datetime(df["data"]).dt.strftime("%Y-%m-%d") == chave[1])
                        row = df[m]
                        if len(row) and pd.isna(row["sent_ia"].iloc[0]) and int(row["n_manchetes"].iloc[0]) >= 3:
                            titles = gdelt_headlines(termo, ws, ws + timedelta(days=7)); time.sleep(args.pausa)
                            if titles:
                                s = sentiment_ai(client, args.modelo, a, titles)
                                df.loc[m, "sent_ia"] = s; df.to_csv(args.saida, index=False)
                                print(f"  {a} {chave[1]}  IA={s}", flush=True)
                    continue
                if args.max_semanas and n_exec >= args.max_semanas:
                    raise KeyboardInterrupt
                titles = gdelt_headlines(termo, ws, ws + timedelta(days=7))
                time.sleep(args.pausa)
                if titles is None:
                    print(f"  {a} {chave[1]}  FALHOU (tentar depois)", flush=True); continue
                lex = sentiment_lexical(titles)
                ia = sentiment_ai(client, args.modelo, a, titles) if (args.ai and len(titles) >= 3) else None
                linha = {"data": ws.strftime("%Y-%m-%d"), "ativo": a, "n_manchetes": len(titles), "sent_lexico": lex, "sent_ia": ia, "fonte": "gdelt"}
                df = pd.concat([df, pd.DataFrame([linha])], ignore_index=True)
                df.to_csv(args.saida, index=False)          # checkpoint a cada semana
                feitos.add(chave); n_exec += 1
                print(f"  {a} {chave[1]}  n={len(titles):2d}  lex={lex}  ia={ia}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrompido — progresso salvo. Rode de novo para continuar.")
    print(f"\nConcluído: {len(df)} linhas em {args.saida}")


if __name__ == "__main__":
    main()
