"""
Histórico DIÁRIO de sentimento de notícias por ativo, usando o modo *timeline* do GDELT DOC 2.0:
uma chamada devolve o tom médio de TODOS os artigos que citam o termo, dia a dia, para um período inteiro.
→ ~1 chamada por ativo por ano (em vez de 1 por semana) — contorna o limite de taxa que trava o coletor semanal.

O "tom" do GDELT é o sentimento calculado sobre o texto completo dos artigos (escala típica −10..+10;
negativo = cobertura negativa). Também coletamos o volume de cobertura (% dos artigos do dia que citam o termo),
que serve como medida de atenção.

Uso:
    python build_sentiment_timeline.py --ativos SOL,BTC,ETH,DOGE,AVAX,ADA,LINK,DOT --anos 5
Saída: sentimento_historico.csv (colunas: data, ativo, tone, volume). Substitui/complementa o CSV semanal:
o app detecta o formato automaticamente. Retomável: pares ativo×ano já coletados são pulados.
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

TERMOS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "DOGE": "dogecoin", "AVAX": '"avalanche" crypto',
    "ADA": "cardano", "LINK": "chainlink", "DOT": "polkadot", "XRP": '"ripple" crypto', "BNB": '"binance coin"',
    "MATIC": '"polygon" crypto', "NEAR": '"near protocol"', "SUI": '"sui" blockchain', "LTC": "litecoin",
    "TRX": '"tron" crypto', "ATOM": '"cosmos" crypto', "UNI": "uniswap", "AAVE": "aave", "ARB": "arbitrum", "OP": '"optimism" crypto',
}
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"


def _timeline(query, mode, start, end, retries=8, pausa=15):
    params = {"query": f"{query} sourcelang:eng", "mode": mode, "format": "json",
              "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S")}
    wait = pausa
    for _ in range(retries):
        try:
            r = requests.get(GDELT, params=params, timeout=60, headers={"User-Agent": "crypto-sentiment-research/1.0"})
            if r.status_code == 200:
                try:
                    j = r.json()
                except ValueError:
                    return {}
                pontos = {}
                for serie in j.get("timeline", []):
                    for d in serie.get("data", []):
                        try:
                            dt = pd.to_datetime(d["date"]).tz_localize(None).normalize() if pd.to_datetime(d["date"]).tzinfo is None \
                                else pd.to_datetime(d["date"]).tz_convert(None).normalize()
                            pontos[dt] = float(d["value"])
                        except (KeyError, ValueError, TypeError):
                            continue
                return pontos
            if r.status_code == 429:
                print(f"    429 — aguardando {wait}s", flush=True); time.sleep(wait); wait = min(wait * 2, 180); continue
            print(f"    HTTP {r.status_code}: {r.text[:120]}", flush=True); time.sleep(wait)
        except requests.RequestException as e:
            print(f"    rede: {e}", flush=True); time.sleep(wait)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ativos", default="SOL,BTC,ETH,DOGE,AVAX,ADA,LINK,DOT")
    ap.add_argument("--anos", type=float, default=5)
    ap.add_argument("--saida", default="sentimento_historico.csv")
    ap.add_argument("--pausa", type=float, default=15, help="segundos entre chamadas")
    ap.add_argument("--sem-volume", action="store_true", help="não coletar a série de volume (metade das chamadas)")
    args = ap.parse_args()

    ativos = [a.strip().upper() for a in args.ativos.split(",") if a.strip()]
    hoje = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    inicio = hoje - timedelta(days=int(args.anos * 365))
    # janelas de ~6 meses: resolução diária garantida e respostas pequenas
    janelas = []
    ini = inicio
    while ini < hoje:
        fim = min(ini + timedelta(days=182), hoje)
        janelas.append((ini, fim)); ini = fim

    if os.path.exists(args.saida):
        df = pd.read_csv(args.saida)
        if "tone" not in df.columns:            # CSV do coletor semanal → arquiva e recomeça no formato diário
            os.replace(args.saida, args.saida.replace(".csv", "_semanal.csv"))
            print(f"CSV semanal antigo movido para {args.saida.replace('.csv', '_semanal.csv')}")
            df = pd.DataFrame(columns=["data", "ativo", "tone", "volume"])
        df["data"] = pd.to_datetime(df["data"], format="mixed", utc=True).dt.tz_localize(None).dt.normalize()
    else:
        df = pd.DataFrame(columns=["data", "ativo", "tone", "volume"])

    print(f"{len(ativos)} ativos × {len(janelas)} janelas | saída: {args.saida}")
    try:
        for a in ativos:
            termo = TERMOS.get(a, f'"{a}" crypto')
            ja = set(pd.to_datetime(df.loc[df["ativo"] == a, "data"]).dt.normalize()) if len(df) else set()
            for ini, fim in janelas:
                dias = pd.date_range(ini, fim - timedelta(days=1), freq="D")
                if len(dias) and all(d in ja for d in dias[:-3]):
                    continue                                   # janela já coletada
                print(f"  {a} {ini.date()} → {fim.date()}", flush=True)
                tone = _timeline(termo, "timelinetone", ini, fim, pausa=args.pausa)
                time.sleep(args.pausa)
                if tone is None:
                    print("    FALHOU (tentar depois)", flush=True); continue
                vol = {}
                if not args.sem_volume:
                    vol = _timeline(termo, "timelinevol", ini, fim, pausa=args.pausa) or {}
                    time.sleep(args.pausa)
                novas = pd.DataFrame([{"data": d, "ativo": a, "tone": t, "volume": vol.get(d)} for d, t in sorted(tone.items())])
                if novas.empty:
                    print("    sem dados nesta janela", flush=True); continue
                df = pd.concat([df[~((df["ativo"] == a) & (df["data"].isin(novas["data"])))], novas], ignore_index=True)
                df.sort_values(["ativo", "data"]).to_csv(args.saida, index=False)
                print(f"    {len(novas)} dias | tom médio {novas['tone'].mean():+.2f}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrompido — progresso salvo.")
    print(f"\nConcluído: {len(df)} linhas em {args.saida}")


if __name__ == "__main__":
    main()
