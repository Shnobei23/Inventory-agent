"""Inventory Decision Agent - BAN 517 Supply Chain Analytics."""
import json

import anthropic
import pandas as pd
import streamlit as st

import inventory_tools as it

st.set_page_config(page_title="Inventory Decision Agent", layout="wide")

MODEL = st.secrets.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
PERIOD = {"type": "string", "enum": ["day", "week", "month", "quarter", "year"]}

TOOLS = [
    {"name": "eoq", "description": "Economic order quantity. Pass demand with its stated period; the function annualizes it.",
     "input_schema": {"type": "object", "properties": {
         "demand": {"type": "number"}, "demand_period": PERIOD,
         "ordering_cost": {"type": "number", "description": "$ per order"},
         "holding_cost": {"type": "number", "description": "$ per unit per holding_cost_period, if given in dollars"},
         "holding_cost_period": PERIOD,
         "holding_rate": {"type": "number", "description": "Holding cost as a decimal fraction of unit cost, e.g. 0.2"},
         "unit_cost": {"type": "number"}},
         "required": ["demand", "demand_period", "ordering_cost"]}},
    {"name": "quantity_discount", "description": "All-units quantity discount: best order quantity and total annual cost including purchase cost.",
     "input_schema": {"type": "object", "properties": {
         "annual_demand": {"type": "number"}, "ordering_cost": {"type": "number"},
         "tiers": {"type": "array", "items": {"type": "object", "properties": {
             "min_qty": {"type": "number", "description": "Smallest quantity that gets this price (0 for the base price)"},
             "unit_price": {"type": "number"}}, "required": ["min_qty", "unit_price"]}},
         "holding_rate": {"type": "number", "description": "Annual holding cost as decimal fraction of unit price"},
         "holding_cost": {"type": "number", "description": "Fixed $ per unit per year, if not a rate"}},
         "required": ["annual_demand", "ordering_cost", "tiers"]}},
    {"name": "safety_stock_rop", "description": "Safety stock and reorder point, including lead-time variability. Pass lead time in its stated unit; the function converts units.",
     "input_schema": {"type": "object", "properties": {
         "avg_demand": {"type": "number"}, "demand_sd": {"type": "number"}, "demand_period": PERIOD,
         "lead_time": {"type": "number"}, "lead_time_sd": {"type": "number", "description": "0 if lead time is constant"},
         "lead_time_unit": PERIOD, "service_level": {"type": "number", "description": "Decimal, e.g. 0.95"}},
         "required": ["avg_demand", "demand_sd", "demand_period", "lead_time", "lead_time_unit", "service_level"]}},
    {"name": "fixed_period_review", "description": "Fixed-period (periodic) review: target level and order quantity.",
     "input_schema": {"type": "object", "properties": {
         "avg_demand": {"type": "number"}, "demand_sd": {"type": "number"}, "demand_period": PERIOD,
         "review_period": {"type": "number"}, "lead_time": {"type": "number"}, "time_unit": PERIOD,
         "service_level": {"type": "number"}, "on_hand": {"type": "number"},
         "on_order": {"type": "number"}, "backorders": {"type": "number"}},
         "required": ["avg_demand", "demand_sd", "demand_period", "review_period", "lead_time", "time_unit",
                      "service_level", "on_hand"]}},
    {"name": "newsvendor", "description": "Single-period newsvendor optimal order quantity with normal demand.",
     "input_schema": {"type": "object", "properties": {
         "selling_price": {"type": "number"}, "unit_cost": {"type": "number"}, "salvage_value": {"type": "number"},
         "mean_demand": {"type": "number"}, "demand_sd": {"type": "number"}, "shortage_penalty": {"type": "number"}},
         "required": ["selling_price", "unit_cost", "salvage_value", "mean_demand", "demand_sd"]}},
    {"name": "inventory_classification", "description": "Multicriteria A/B/C/D classification of the uploaded CSV (Sales 65%, Hits 35%). Returns the summary, top items per class, and optional item details.",
     "input_schema": {"type": "object", "properties": {
         "item_numbers": {"type": "array", "items": {"type": "integer"}, "description": "Items to look up (optional)"},
         "top_n": {"type": "integer", "description": "Top items by Sales per class (default 5)"}}}},
]

SYSTEM = """You are an Inventory Decision Agent for a supply chain analytics course.
Rules:
- NEVER calculate numbers yourself. Always call the matching tool and report only the numbers it returns.
- Extract inputs exactly as stated, including their time units. Do not convert units yourself: pass the stated period/unit to the tool, which converts them.
- Percentages go to tools as decimals (95% -> 0.95, 20% -> 0.20).
- For quantity discounts, give each price tier's minimum quantity (base price min_qty = 0).
Format every answer with these headings:
**Extracted Inputs** (with units), **Unit Conversions & Assumptions**, **Formula**, **Results** (a table where helpful; for quantity discounts include the candidate table), **Recommendation** (2-3 sentences, practical).
If required inputs are missing, ask for them instead of guessing."""

FUNCS = {"eoq": it.eoq, "quantity_discount": it.quantity_discount, "safety_stock_rop": it.safety_stock_rop,
         "fixed_period_review": it.fixed_period_review, "newsvendor": it.newsvendor}


def run_tool(name, args):
    if name == "inventory_classification":
        if st.session_state.get("df") is None:
            return {"error": "No CSV uploaded. Ask the user to upload the inventory CSV in the sidebar."}
        detail, summary = it.classify_inventory(st.session_state["df"])
        n = args.get("top_n", 5)
        top = {c: detail[detail["Final Class"] == c].head(n)[["Item Number", "Item Name", "Sales ($)"]]
               .to_dict("records") for c in "ABCD"}
        items = detail[detail["Item Number"].isin(args.get("item_numbers", []))].to_dict("records")
        return {"tool": "Multicriteria Inventory Classification", "summary": summary.reset_index().to_dict("records"),
                f"top_{n}_by_sales": top, "item_details": items}
    return FUNCS[name](**{k: v for k, v in args.items() if v is not None})


def ask_agent(history):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    messages, calls = list(history), []
    for _ in range(6):
        resp = client.messages.create(model=MODEL, max_tokens=2500, system=SYSTEM, tools=TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text"), calls
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                try:
                    out = run_tool(b.name, b.input)
                except Exception as e:  # report errors back to the model
                    out = {"error": str(e)}
                calls.append({"tool": b.name, "input": b.input, "output": out})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=str)})
        messages.append({"role": "user", "content": results})
    return "Stopped after too many tool calls.", calls


# ---------------------------------------------------------------- UI
st.title("Inventory Decision Agent")
st.caption("EOQ | Quantity Discount | Safety Stock & ROP | Fixed-Period Review | Newsvendor | Multicriteria Classification")

with st.sidebar:
    st.header("Inventory data")
    up = st.file_uploader("Upload inventory CSV", type="csv")
    if up is not None:
        st.session_state["df"] = pd.read_csv(up, encoding="utf-8")
        st.success(f"Loaded {len(st.session_state['df'])} rows")
        try:  # add classification columns to the uploaded dataframe (UoM kept as a display column)
            st.session_state["df_classified"], st.session_state["class_summary"] = it.attach_classification(
                st.session_state["df"])
            st.session_state["dropped_rows"] = it.find_dropped_rows(st.session_state["df"])
        except Exception as e:
            st.session_state["df_classified"] = st.session_state["class_summary"] = None
            st.session_state["dropped_rows"] = None
            st.error(f"Could not classify this file: {e}")
        if st.session_state.get("dropped_rows") is not None and len(st.session_state["dropped_rows"]):
            st.warning(f"{len(st.session_state['dropped_rows'])} row(s) excluded from classification "
                       "(Sales or Hits not a number). See the Inventory Classification tab.")

tab_agent, tab_class = st.tabs(["Ask the Agent", "Inventory Classification"])

with tab_agent:
    st.session_state.setdefault("chat", [])
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if prompt := st.chat_input("Describe an inventory problem, e.g. 'Find EOQ. Demand is 1,000 units per month...'"):
        st.session_state["chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Calculating..."):
                answer, calls = ask_agent([{"role": m["role"], "content": m["content"]} for m in st.session_state["chat"]])
            st.markdown(answer)
            for c in calls:
                with st.expander(f"Tool call: {c['tool']} (inputs and exact outputs)"):
                    st.json({"input": c["input"], "output": c["output"]})
        st.session_state["chat"].append({"role": "assistant", "content": answer})

with tab_class:
    st.subheader("Multicriteria Inventory Classification")
    st.markdown("Sales and Hits are classified separately by cumulative % of each total "
                "(A <= 60%, B <= 80%, C <= 90%, D > 90%), scored A=4, B=3, C=2, D=1. "
                "**Combined Score = 0.65 x Sales Score + 0.35 x Hits Score**. "
                "Final class: A >= 3.25, B >= 2.50, C >= 1.75, D < 1.75.")
    if st.session_state.get("df") is None:
        st.info("Upload the inventory CSV in the sidebar.")
    elif st.session_state.get("df_classified") is None:
        st.warning("The uploaded file could not be classified. Check that it has Item Number, Sales ($) and Hits columns.")
    else:
        detail, summary = st.session_state["df_classified"], st.session_state["class_summary"]
        dropped = st.session_state.get("dropped_rows")
        if dropped is not None and len(dropped):
            st.warning(f"{len(dropped)} row(s) could not be classified because Sales ($) or Hits is not a number, "
                       f"so they are excluded from every count and total below ({len(st.session_state['df'])} rows "
                       f"uploaded, {len(detail)} classified).")
            st.dataframe(dropped, hide_index=True, width="stretch")
        st.markdown("**Inputs used**")
        st.table(pd.DataFrame({"Input": ["Items classified", "Criteria columns", "Weights", "Cumulative % cutoffs",
                                         "Scores", "Final class cutoffs", "Total Sales ($)", "Total Hits"],
                               "Value": [f"{len(detail):,} items (key: Item Number)",
                                         "Sales ($) and Hits, each ranked independently, descending",
                                         "Sales 65%, Hits 35%", "A <= 60%, B <= 80%, C <= 90%, D > 90%",
                                         "A = 4, B = 3, C = 2, D = 1",
                                         "Combined Score: A >= 3.25, B >= 2.50, C >= 1.75, D < 1.75",
                                         f"${detail['Sales ($)'].sum():,.2f}", f"{detail['Hits'].sum():,.0f}"]}
                              ).set_index("Input"))
        st.markdown("**Classification summary**")
        st.dataframe(summary.style.format({"Number of Items": "{:,.0f}", "Total Sales ($)": "${:,.2f}",
                                           "% of Total Sales": "{:.2f}%"}), width="stretch")
        st.markdown("**Top 5 items by Sales in each final class**")
        cols = st.columns(4)
        for col, c in zip(cols, "ABCD"):
            col.markdown(f"Class {c}")
            col.dataframe(detail[detail["Final Class"] == c].head(5)[["Item Number", "Item Name", "Sales ($)"]],
                          hide_index=True, width="stretch")
        look = st.number_input("Look up an item number", value=1170, step=1)
        st.dataframe(detail[detail["Item Number"] == look], hide_index=True, width="stretch")
        flt = st.multiselect("Filter by final class", list("ABCD"), default=list("ABCD"))
        st.markdown("**All items (with intermediate calculations)**")
        st.dataframe(detail[detail["Final Class"].isin(flt)], hide_index=True, width="stretch")
        st.download_button("Download classification CSV", detail.to_csv(index=False), "inventory_classification.csv")
