import os
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Tuple, Any, Dict
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Load env
load_dotenv()

# Import Google ADK Agent (strict usage). Install google-adk per README if missing.
from google.adk.agents import Agent  # assumes google.adk is installed and exposes Agent

# --- TOOL IMPLEMENTATIONS ---


def _normalize_ticker(raw: str) -> str:
    raw = raw.strip().upper()
    # Basic mapping for common Indian names -> NSE tickers
    indian_map = {
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFY": "INFY.NS",
        "ZOMATO": "ZOMATO.NS",
        "RELIANCE INDUSTRIES": "RELIANCE.NS",
        "INFOSYS": "INFY.NS",
    }
    if raw in indian_map:
        return indian_map[raw]
    # if already contains exchange suffix or looks like ticker
    if "." in raw:
        return raw
    # fallback: return as-is (yfinance will handle many tickers)
    return raw


def get_stock_price(ticker: str) -> Dict[str, Any]:
    t = _normalize_ticker(ticker)
    try:
        tk = yf.Ticker(t)
        info = tk.info or {}
        return {
            "ticker": t,
            "current_price": info.get("currentPrice"),
            "previous_close": info.get("previousClose"),
            "currency": info.get("currency", "USD"),
            "market_state": info.get("marketState"),
            "raw_info_keys": list(info.keys())[:20],
        }
    except Exception as e:
        return {"error": str(e), "ticker": t}


def get_historical_data(ticker: str, period: str = "1mo", interval: str = "1d") -> Dict[str, Any]:
    t = _normalize_ticker(ticker)
    try:
        tk = yf.Ticker(t)
        hist = tk.history(period=period, interval=interval)
        if hist.empty:
            return {"error": "no_data", "message": f"No historical data for {t}", "ticker": t}
        # Return last 90 rows max and the DataFrame for plotting
        df = hist.reset_index()
        return {"ticker": t, "dataframe": df, "summary": df.tail(5).to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e), "ticker": t}


def get_stock_fundamentals(ticker: str) -> Dict[str, Any]:
    t = _normalize_ticker(ticker)
    try:
        tk = yf.Ticker(t)
        info = tk.info or {}
        return {
            "ticker": t,
            "PE_Ratio": info.get("trailingPE"),
            "Market_Cap": info.get("marketCap"),
            "EPS": info.get("trailingEps"),
            "Dividend_Yield": info.get("dividendYield"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception as e:
        return {"error": str(e), "ticker": t}


def get_stock_news(ticker: str, limit: int = 3) -> Dict[str, Any]:
    """
    Robust news fetcher. Ensure local var names are correct (no UnboundLocalError).
    Returns dict {"ticker": normalized_ticker, "news": [ {title, link, publisher}, ... ] }
    """
    t = _normalize_ticker(ticker)
    try:
        items = []
        # 1) Try structured yfinance news first (filter out empty entries)
        try:
            tk = yf.Ticker(t)
            news_items = getattr(tk, "news", None) or []
            for n in news_items:
                title = n.get("title") or n.get("headline")
                link = n.get("link") or n.get("url")
                publisher = n.get("publisher") or n.get("source")
                if title and link:
                    if link.startswith("/"):
                        link = "https://finance.yahoo.com" + link
                    items.append({"title": title.strip(), "link": link, "publisher": publisher})
                if len(items) >= limit:
                    break
        except Exception:
            items = items or []

        # 2) Try ticker without exchange suffix (e.g., RELIANCE)
        if not items and "." in t:
            alt = t.split(".")[0]
            try:
                tk2 = yf.Ticker(alt)
                news_items2 = getattr(tk2, "news", None) or []
                for n in news_items2:
                    title = n.get("title") or n.get("headline")
                    link = n.get("link") or n.get("url")
                    publisher = n.get("publisher") or n.get("source")
                    if title and link:
                        if link.startswith("/"):
                            link = "https://finance.yahoo.com" + link
                        items.append({"title": title.strip(), "link": link, "publisher": publisher})
                    if len(items) >= limit:
                        break
            except Exception:
                pass

        # 3) Scrape Yahoo / Google News RSS as fallbacks (keeps previous logic)
        if not items:
            alt_candidates = [t]
            if "." in t:
                alt_candidates.append(t.split(".")[0])
            urls = []
            for cand in alt_candidates:
                urls.extend([
                    f"https://finance.yahoo.com/quote/{cand}/news",
                    f"https://in.finance.yahoo.com/quote/{cand}/news",
                    f"https://finance.yahoo.com/quote/{cand}",
                ])
            seen_links = set()
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            selectors = [
                "h3 a",
                "li.js-stream-content a",
                "article a",
                "a[href*='/news/']",
                "a[href*='/story/']",
                "a[href*='/article/']",
            ]
            for url in urls:
                try:
                    resp = requests.get(url, headers=headers, timeout=8)
                    if resp.status_code != 200:
                        continue
                    soup = BeautifulSoup(resp.text, "html.parser")
                    found = []
                    for sel in selectors:
                        for a in soup.select(sel):
                            title = a.get_text(strip=True)
                            link = a.get("href") or a.get("data-href")
                            if not title or not link:
                                continue
                            if link.startswith("/"):
                                link = "https://finance.yahoo.com" + link
                            if link in seen_links:
                                continue
                            seen_links.add(link)
                            found.append({"title": title, "link": link, "publisher": None})
                            if len(found) >= limit:
                                break
                        if len(found) >= limit:
                            break
                    if found:
                        items.extend(found)
                    if len(items) >= limit:
                        break
                except Exception:
                    time.sleep(0.5)
                    continue

        # 4) Google News RSS fallback (no API key required)
        if not items:
            # try both ticker and plain company name
            rss_queries = [t]
            if "." in t:
                rss_queries.append(t.split(".")[0])
            for q in rss_queries:
                try:
                    rss_q = quote_plus(f"{q} stock OR {q} shares OR {q} company")
                    rss_url = f"https://news.google.com/rss/search?q={rss_q}&hl=en-IN&gl=IN&ceid=IN:en"
                    resp = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                    if resp.status_code != 200 or not resp.text:
                        continue
                    root = ET.fromstring(resp.text)
                    channel = root.find("channel")
                    if channel is None:
                        continue
                    found = []
                    for item in channel.findall("item"):
                        title_el = item.find("title")
                        link_el = item.find("link")
                        source_el = item.find("{http://www.w3.org/2005/Atom}source") or item.find("source")
                        title = title_el.text.strip() if title_el is not None and title_el.text else None
                        link = link_el.text.strip() if link_el is not None and link_el.text else None
                        publisher = source_el.text.strip() if source_el is not None and source_el.text else None
                        if title and link:
                            found.append({"title": title, "link": link, "publisher": publisher})
                        if len(found) >= limit:
                            break
                    if found:
                        items.extend(found)
                    if len(items) >= limit:
                        break
                except Exception:
                    continue

        # final: dedupe and trim
        unique = []
        seen = set()
        for it in items:
            key = (it.get("title") or "") + "|" + (it.get("link") or "")
            if key in seen:
                continue
            seen.add(key)
            unique.append(it)
            if len(unique) >= limit:
                break

        return {"ticker": t, "news": unique}
    except Exception as e:
        return {"error": str(e), "ticker": t}


def compare_stocks(tickers: List[str], period: str = "1mo") -> Dict[str, Any]:
    results = {}
    for raw in tickers:
        t = _normalize_ticker(raw)
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period=period)
            if hist.empty:
                results[t] = {"error": "no_data"}
                continue
            start = hist["Close"].iloc[0]
            end = hist["Close"].iloc[-1]
            pct = ((end - start) / start) * 100 if start and start != 0 else None
            results[t] = {"start_close": start, "end_close": end, "percent_change": pct}
        except Exception as e:
            results[t] = {"error": str(e)}
    return {"period": period, "results": results}


# --- AGENT CONFIGURATION & CREATION ---


SYSTEM_INSTRUCTION = """
You are StockInsightsAgent. Use the provided tools (get_stock_price, get_historical_data, 
get_stock_fundamentals, compare_stocks, get_stock_news) to answer user queries. 
Always call tools for factual data. Do not hallucinate figures. Return concise output.
"""

def create_stock_agent(api_key: Optional[str] = None) -> Agent:
    """
    Create and return a Google ADK Agent configured with tools.
    If api_key is provided, set it into the environment instead of passing to Agent.
    """
    # prefer explicit arg, else read from env
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing in environment")

    # ensure ADK/LLM picks up the key from environment (do NOT pass as kwarg to Agent)
    os.environ["GOOGLE_API_KEY"] = api_key

    # Wrap tools in simple callables the ADK agent can use.
    tools = [
        get_stock_price,
        get_historical_data,
        get_stock_fundamentals,
        compare_stocks,
        get_stock_news,
    ]

    agent = Agent(
        name="StockInsightsAgent",
        model="gemini-3-small",  # change as needed
        instruction=SYSTEM_INSTRUCTION,
        tools=tools,
        # removed api_key parameter (Agent does not accept it)
    )
    return agent


# --- LOCAL TOOL SELECTOR (fallback/fast route) ---


def detect_intent(query: str) -> str:
    q = query.lower()
    if "compare" in q:
        return "compare"
    if any(k in q for k in ["price", "current price", "quote"]):
        return "price"
    if any(k in q for k in ["history", "historical", "trend", "chart", "plot"]):
        return "history"
    if any(k in q for k in ["pe ratio", "eps", "market cap", "fundamental", "fundamentals"]):
        return "fundamentals"
    if "news" in q:
        return "news"
    return "general"


def parse_tickers_from_text(text: str) -> List[str]:
    """
    Extract probable tickers/company names from a user query.
    - Ignores common stop words (e.g. WHAT, IS, THE, PRICE)
    - Maps known company names to NSE tickers
    - Preserves order and uniqueness
    """
    text_up = text.upper()
    # tokens that should not be considered as tickers
    STOP_WORDS = {
        "WHAT", "IS", "THE", "PLEASE", "SHOW", "GIVE", "ME", "TELL", "PRICE", "OF", "FOR",
        "AND", "VS", "COMPARE", "STOCK", "STOCKS", "QUOTE", "HISTORY", "HISTORICAL",
        "TREND", "CHART", "PLOT", "NEWS", "A", "AN", "ON", "IN", "ABOUT"
    }

    # Mapping common names -> tickers (extend as needed)
    KNOWN_MAP = {
        "RELIANCE": "RELIANCE.NS",
        "RELIANCE INDUSTRIES": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFOSYS": "INFY.NS",
        "INFY": "INFY.NS",
        "ZOMATO": "ZOMATO.NS",
        "TESLA": "TSLA",
        "APPLE": "AAPL",
        "MICROSOFT": "MSFT",
        "GOOGLE": "GOOGL",
        "ALPHABET": "GOOGL",
    }

    # find uppercase-like tokens (tickers or words)
    tokens = re.findall(r'\b[A-Z\.]{1,12}\b', text_up)
    candidates: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in STOP_WORDS:
            i += 1
            continue

        # try two-word match (e.g., "RELIANCE INDUSTRIES")
        two = None
        if i + 1 < len(tokens):
            two = f"{t} {tokens[i+1]}"
            if two in KNOWN_MAP:
                candidates.append(_normalize_ticker(KNOWN_MAP[two]))
                i += 2
                continue

        # single token known name
        if t in KNOWN_MAP:
            candidates.append(_normalize_ticker(KNOWN_MAP[t]))
        # plausible ticker-like token (letters/dots, short)
        elif re.match(r'^[A-Z\.]{1,6}$', t):
            candidates.append(_normalize_ticker(t))
        # otherwise skip
        i += 1

    # preserve order and uniqueness
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result