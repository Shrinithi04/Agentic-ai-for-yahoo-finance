import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

from agent_logic import (
    create_stock_agent,
    detect_intent,
    parse_tickers_from_text,
    get_stock_price,
    get_historical_data,
    get_stock_fundamentals,
    get_stock_news,
    compare_stocks,
)

st.set_page_config(page_title="Stock Insights AI Agent", layout="wide")

st.title("Stock Insights — AI Agent (yfinance + Google ADK + Streamlit)")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment. Add it to .env as GOOGLE_API_KEY.")
    st.stop()

try:
    agent = create_stock_agent(api_key)
except Exception as e:
    st.error(f"Failed to create ADK Agent: {e}")
    st.stop()

# --- removed diagnostics (temporary) ---

with st.sidebar:
    st.header("Quick examples")
    st.markdown("- What is the price of TCS?\n- Show Tesla trend for 5 days\n- Compare AAPL and MSFT\n- Give news about Reliance\n")
    period = st.selectbox("Default historical period", ["5d", "1mo", "6mo", "1y"], index=1)

# small helper: format numeric values (handles numpy scalars)
def _fmt_num(v):
    try:
        if hasattr(v, "item"):
            v = v.item()
        v = float(v)
        if abs(v) >= 1_000_000_000:
            return f"{v:,.0f}"
        if abs(v) >= 1_000_000:
            return f"{v:,.0f}"
        return f"{v:,.2f}"
    except Exception:
        return str(v)

# format price in plain English (no company info list)
def format_price_readable(d: dict) -> str:
    ticker = d.get("ticker", "N/A")
    cur = d.get("current_price")
    prev = d.get("previous_close")
    currency = d.get("currency", "")
    market = d.get("market_state", "")

    sym = "₹" if str(currency).upper() == "INR" else ""

    pct = None
    try:
        if cur is not None and prev is not None:
            curf = float(cur)
            prevf = float(prev)
            if prevf != 0:
                pct = (curf - prevf) / prevf * 100
    except Exception:
        pct = None

    lines = []
    lines.append(f"{ticker} — Price")
    lines.append(f"Current price: {sym}{_fmt_num(cur)}")
    lines.append(f"Previous close: {sym}{_fmt_num(prev)}")
    if pct is not None:
        sign = "+" if pct >= 0 else ""
        lines.append(f"Change vs previous close: {sign}{pct:.2f}%")
    lines.append(f"Currency: {currency} • Market state: {market}")
    return "\n".join(lines)

# serialize tool outputs into a compact plain text summary that the LLM can use as context
def serialize_tool_output_for_llm(intent: str, tool_output: dict) -> str:
    if not isinstance(tool_output, dict):
        return str(tool_output)

    if intent == "price":
        return (
            f"Ticker: {tool_output.get('ticker')}\n"
            f"Current price: {tool_output.get('current_price')}\n"
            f"Previous close: {tool_output.get('previous_close')}\n"
            f"Currency: {tool_output.get('currency')}\n"
            f"Market state: {tool_output.get('market_state')}\n"
        )

    if intent == "history":
        df = tool_output.get("dataframe")
        # give a tiny summary to LLM (last and first close)
        if df is not None and hasattr(df, "shape"):
            try:
                first = df.iloc[0]
                last = df.iloc[-1]
                return (
                    f"Ticker: {tool_output.get('ticker')}\n"
                    f"First date close: {first.get('Close')} (index 0)\n"
                    f"Last date close: {last.get('Close')} (index {len(df)-1})\n"
                )
            except Exception:
                return str(tool_output)
        return str(tool_output)

    if intent == "fundamentals":
        return (
            f"Ticker: {tool_output.get('ticker')}\n"
            f"PE ratio: {tool_output.get('PE_Ratio')}\n"
            f"Market cap: {tool_output.get('Market_Cap')}\n"
            f"EPS: {tool_output.get('EPS')}\n"
        )

    if intent == "compare":
        results = tool_output.get("results", {})
        lines = [f"Comparison period: {tool_output.get('period')}"]
        for tk, stats in (results or {}).items():
            if "error" in stats:
                lines.append(f"{tk}: error {stats['error']}")
            else:
                pct = stats.get("percent_change")
                lines.append(f"{tk}: {pct:+.2f}% change")
        return "\n".join(lines)

    if intent == "news":
        items = tool_output.get("news", []) or []
        lines = []
        for it in items:
            title = it.get("title") or "No title"
            link = it.get("link") or ""
            lines.append(f"- {title} {link}")
        return "\n".join(lines) if lines else "No news items found."

    # fallback
    return str(tool_output)

def send_to_agent(agent, prompt: str):
    """
    More resilient caller for Google ADK agent objects.
    Tries: direct call if agent is callable, common method names, and multiple arg shapes.
    """
    if agent is None:
        raise RuntimeError("No agent available")

    last_exc = None

    # 0) If the agent object itself is callable, try it first
    if callable(agent):
        try:
            return agent(prompt)
        except Exception as e:
            last_exc = e
            # try alternative call shapes
            try:
                return agent({"input": prompt})
            except Exception as e2:
                last_exc = e2
            try:
                return agent(messages=[{"role": "user", "content": prompt}])
            except Exception as e3:
                last_exc = e3

    # 1) Try common method names with several call signatures
    method_names = (
        "run",
        "chat",
        "chat_completion",
        "create_chat",
        "generate",
        "complete",
        "predict",
        "invoke",
        "call",
        "respond",
        "send",
    )
    for name in method_names:
        fn = getattr(agent, name, None)
        if not callable(fn):
            continue
        for attempt in (
            (prompt,),
            ({"input": prompt},),
            ({"prompt": prompt},),
            ({"messages": [{"role": "user", "content": prompt}]},),
            ( [{"role": "user", "content": prompt}], ),
        ):
            try:
                return fn(*attempt)
            except TypeError as e:
                last_exc = e
                continue
            except Exception as e:
                last_exc = e
                continue

    # 2) Try inspecting for a 'client' or 'llm' attribute that might be callable
    for attr in ("client", "llm", "model", "_client"):
        sub = getattr(agent, attr, None)
        if callable(sub):
            try:
                return sub(prompt)
            except Exception as e:
                last_exc = e
        elif sub is not None:
            # try common methods on the sub-object
            for name in method_names:
                fn = getattr(sub, name, None)
                if not callable(fn):
                    continue
                try:
                    return fn(prompt)
                except Exception as e:
                    last_exc = e

    # If nothing worked, raise with a hint to dump diagnostics
    raise RuntimeError(
        "Agent has no supported callable method. Tried common call patterns. "
        "Run diagnostics (see below) and paste the output if you need further help. "
        f"Last error: {repr(last_exc)}"
    )

def get_llm_context(agent, intent: str, tool_output: dict, user_query: str) -> str:
    """
    Try to get context from the ADK agent; if the agent is not invokable,
    return a plain-English fallback explanation built from the tool output.
    """
    # Compose prompt (used only when agent callable)
    summary = serialize_tool_output_for_llm(intent, tool_output)
    prompt = (
        f"User question: {user_query}\n\n"
        f"Tool output summary:\n{summary}\n\n"
        "Provide a concise, plain-English explanation and context for the user based on this data. "
        "Do not include JSON, do not repeat raw data verbatim, and keep the answer focused and actionable."
    )

    # Try ADK agent first
    try:
        response = send_to_agent(agent, prompt)
        if isinstance(response, dict):
            return response.get("text") or response.get("content") or str(response)
        text = getattr(response, "text", None)
        return text if text else str(response)
    except Exception:
        # Fallback: produce a deterministic plain-English summary (no LLM)
        try:
            if not isinstance(tool_output, dict):
                return str(tool_output)

            if intent == "price":
                t = tool_output.get("ticker", "the ticker")
                cur = tool_output.get("current_price")
                prev = tool_output.get("previous_close")
                currency = tool_output.get("currency", "")
                sym = "₹" if str(currency).upper() == "INR" else ""
                try:
                    curf = float(cur) if cur is not None else None
                    prevf = float(prev) if prev is not None else None
                    if curf is not None and prevf is not None and prevf != 0:
                        pct = (curf - prevf) / prevf * 100
                        trend = ("up" if pct > 0 else "down") + f" {abs(pct):.2f}% vs previous close"
                    else:
                        trend = "no change information available"
                except Exception:
                    trend = "no change information available"
                return (
                    f"{t} is currently at {sym}{_fmt_num(cur)}. "
                    f"Previous close was {sym}{_fmt_num(prev)}, {trend}. "
                    "This is factual market data; consider checking the recent trend and fundamentals before making decisions."
                )

            if intent == "compare":
                period = tool_output.get("period", "the selected period")
                results = tool_output.get("results", {}) or {}
                if not results:
                    return "No comparison data available."
                lines = [f"Comparison over {period}:"]
                for tk, stats in results.items():
                    if "error" in stats:
                        lines.append(f"{tk}: data not available")
                        continue
                    pct = stats.get("percent_change")
                    pct_s = f"{float(pct):+.2f}%" if pct is not None else "N/A"
                    lines.append(f"{tk}: changed {pct_s}")
                lines.append("Higher percent means better performance over the period. Use this as a quick signal, not investment advice.")
                return " ".join(lines)

            if intent == "news":
                items = tool_output.get("news", []) or []
                if not items:
                    return f"No recent news found for {tool_output.get('ticker') or 'the ticker'}."
                first = items[0]
                title = first.get("title") or "No title"
                pub = first.get("publisher") or ""
                return f"Latest headline: {title}" + (f" ({pub})." if pub else ".")

            if intent == "fundamentals":
                t = tool_output.get("ticker", "the ticker")
                pe = tool_output.get("PE_Ratio")
                mc = tool_output.get("Market_Cap")
                eps = tool_output.get("EPS")
                return (
                    f"{t} fundamentals: PE ratio { _fmt_num(pe) }, market cap { _fmt_num(mc) }, "
                    f"EPS { _fmt_num(eps) }. Use these metrics to compare valuation vs peers."
                )

            if intent == "history":
                df = tool_output.get("dataframe")
                if df is not None and hasattr(df, "shape") and len(df) >= 2:
                    try:
                        first = float(df["Close"].iloc[0])
                        last = float(df["Close"].iloc[-1])
                        pct = (last - first) / first * 100 if first != 0 else None
                        trend = f"up {pct:.2f}%" if pct and pct > 0 else (f"down {abs(pct):.2f}%" if pct else "no change")
                        return f"Over the selected period the close price moved from {_fmt_num(first)} to {_fmt_num(last)} ({trend}). Check the chart for daily movement."
                    except Exception:
                        pass
                return "Historical data available — view the chart for detailed trend."

            # fallback generic
            # build short key:value lines
            parts = []
            for k, v in tool_output.items():
                if k == "dataframe":
                    parts.append(f"{k}: <dataframe with {len(v) if v is not None else 0} rows>")
                else:
                    if hasattr(v, "item"):
                        try:
                            v = v.item()
                        except Exception:
                            pass
                    parts.append(f"{k}: {v}")
            return " ".join(parts)
        except Exception as e:
            return f"(No LLM available; fallback generation failed: {e})"
# Centralized display: show the tool output readable text once, then LLM context
def display_tool_and_context(agent, intent: str, tool_output: dict, user_query: str):
    """
    Render tool output (human-readable) once and then request LLM context (or fallback).
    Must be defined before it's called by the submit handler.
    """
    try:
        # Price
        if intent == "price":
            try:
                st.text(format_price_readable(tool_output))
            except Exception:
                st.text(str(tool_output))

        # History/chart
        elif intent == "history":
            try:
                df = tool_output.get("dataframe") if isinstance(tool_output, dict) else None
                if df is not None:
                    fig = px.line(df, x=df.columns[0], y="Close", title=f"{tool_output.get('ticker')} — Close Price")
                    st.plotly_chart(fig, use_container_width=True)
                    st.text(f"Showing historical data for {tool_output.get('ticker')}.")
                else:
                    st.text(str(tool_output))
            except Exception:
                st.text(str(tool_output))

        # Fundamentals
        elif intent == "fundamentals":
            try:
                lines = []
                lines.append(f"{tool_output.get('ticker')} — Fundamentals")
                lines.append(f"PE ratio: {_fmt_num(tool_output.get('PE_Ratio'))}")
                lines.append(f"Market cap: {_fmt_num(tool_output.get('Market_Cap'))}")
                lines.append(f"EPS: {_fmt_num(tool_output.get('EPS'))}")
                st.text("\n".join(lines))
            except Exception:
                st.text(str(tool_output))

        # Compare
        elif intent == "compare":
            try:
                st.text(format_compare_readable(tool_output) if isinstance(tool_output, dict) else str(tool_output))
            except Exception:
                st.text(str(tool_output))

        # News
        elif intent == "news":
            try:
                items = tool_output.get("news", []) or []
                if not items:
                    st.text(f"No recent news found for {tool_output.get('ticker')}.")
                else:
                    st.text(f"Latest news for {tool_output.get('ticker')}:")
                    for it in items:
                        title = it.get("title") or "No title"
                        pub = it.get("publisher") or ""
                        link = it.get("link") or ""
                        st.text(f"- {title}" + (f" ({pub})" if pub else ""))
                        if link:
                            st.text(f"  {link}")
            except Exception:
                st.text(str(tool_output))

        else:
            st.text(str(tool_output))

        # LLM context (or deterministic fallback)
        if agent is not None:
            ctx = get_llm_context(agent, intent, tool_output, user_query)
            st.markdown("**Context / Explanation (from LLM or fallback):**")
            st.text(ctx)
    except Exception as e:
        st.text(f"(display error) {e}")
        st.text(str(tool_output))
# reuse earlier format_compare_readable from file (if not present, define quickly)
def format_compare_readable(d: dict) -> str:
    period = d.get("period", "the requested period")
    results = d.get("results", {}) or {}
    lines = [f"Comparison over {period}:"]
    if not results:
        lines.append("No comparison data available.")
        return "\n".join(lines)
    for tk, stats in results.items():
        if "error" in stats:
            lines.append(f"- {tk}: error - {stats['error']}")
            continue
        start = stats.get("start_close")
        end = stats.get("end_close")
        pct = stats.get("percent_change")
        try:
            start_s = _fmt_num(start)
            end_s = _fmt_num(end)
            pct_f = float(pct) if pct is not None else None
            # show delta as absolute value and percent
            if pct_f is not None:
                delta = end - start
                lines.append(f"- {tk}: {start_s} ➔ {end_s} ({"+" if delta >= 0 else ""}{delta} / {pct_f:+.2f}%)")
            else:
                lines.append(f"- {tk}: {start_s} ➔ {end_s} (no percent change data)")
        except Exception:
            lines.append(f"- {tk}: {stats.get('error', 'unknown error')}")
    return "\n".join(lines)

# --- place the form/submit-handling AFTER helpers and display_tool_and_context definitions ---
with st.form("query_form"):
    query = st.text_input("Ask a stock question or command", value="What is the price of AAPL?")
    submit = st.form_submit_button("Ask")

# handle submission inside a try/except and show spinner/errors
if submit:
    if not query or not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Working..."):
            try:
                intent = detect_intent(query)
                tickers = parse_tickers_from_text(query)

                handlers = {
                    "price": lambda: get_stock_price(tickers[0]) if tickers else {"error": "No ticker found"},
                    "history": lambda: get_historical_data(tickers[0], period=period) if tickers else {"error": "No ticker found"},
                    "fundamentals": lambda: get_stock_fundamentals(tickers[0]) if tickers else {"error": "No ticker found"},
                    "news": lambda: get_stock_news(tickers[0], limit=5) if tickers else {"error": "No ticker found"},
                    "compare": lambda: compare_stocks(tickers, period=period) if tickers and len(tickers) >= 2 else {"error": "Need at least two tickers to compare"},
                    "general": lambda: send_to_agent(agent, query) if agent else {"error": "No agent available"},
                }
                handler = handlers.get(intent, handlers["general"])
                tool_result = handler()

                if intent == "general":
                    text = tool_result.get("text") if isinstance(tool_result, dict) else getattr(tool_result, "text", None)
                    st.text(text if text else str(tool_result))
                else:
                    display_tool_and_context(agent, intent, tool_result, query)
            except Exception as e:
                import traceback
                st.error(f"Error while processing the request: {e}")
                st.text(traceback.format_exc())