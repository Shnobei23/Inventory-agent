"""Deterministic inventory-analytics functions for the Inventory Decision Agent.

The LLM never computes numbers: it extracts inputs and calls these functions.
All unit conversions happen here, so they are explicit and testable.
"""
import math

import pandas as pd
from scipy.stats import norm

PERIODS_PER_YEAR = {"day": 365, "week": 52, "month": 12, "quarter": 4, "year": 1}
DAYS_PER = {"day": 1, "week": 7, "month": 365 / 12, "quarter": 365 / 4, "year": 365}


def _r(x, n=2):
    return round(float(x), n)


# ---------------------------------------------------------------- EOQ
def eoq(demand, demand_period, ordering_cost, holding_cost=None, holding_cost_period="year",
        holding_rate=None, unit_cost=None):
    """EOQ = sqrt(2DS/H) with D and H both annual."""
    conversions = []
    D = demand * PERIODS_PER_YEAR[demand_period]
    if demand_period != "year":
        conversions.append(f"Annual demand D = {demand:,} per {demand_period} x "
                           f"{PERIODS_PER_YEAR[demand_period]} = {D:,.2f} units/year")
    if holding_cost is None:
        if holding_rate is None or unit_cost is None:
            raise ValueError("Provide holding_cost, or holding_rate and unit_cost.")
        H = holding_rate * unit_cost
        conversions.append(f"H = {holding_rate:.0%} x ${unit_cost:,.2f} = ${H:,.4f} per unit per year")
    else:
        H = holding_cost * PERIODS_PER_YEAR[holding_cost_period]
        if holding_cost_period != "year":
            conversions.append(f"Annual H = ${holding_cost} per {holding_cost_period} x "
                               f"{PERIODS_PER_YEAR[holding_cost_period]} = ${H:,.4f} per unit per year")
    S = ordering_cost
    Q = math.sqrt(2 * D * S / H)
    orders = D / Q
    return {
        "tool": "EOQ",
        "inputs": {"Annual demand D (units/yr)": _r(D), "Ordering cost S ($/order)": _r(S),
                   "Holding cost H ($/unit/yr)": _r(H, 4)},
        "conversions": conversions,
        "formula": "EOQ = sqrt(2DS / H); Orders/yr = D/Q; Annual ordering cost = (D/Q)S; "
                   "Annual holding cost = (Q/2)H",
        "results": {"EOQ (units)": _r(Q), "Orders per year": _r(orders),
                    "Time between orders (days)": _r(365 / orders),
                    "Annual ordering cost ($)": _r(orders * S), "Annual holding cost ($)": _r(Q / 2 * H),
                    "Total annual ordering + holding cost ($)": _r(orders * S + Q / 2 * H)},
    }


# ---------------------------------------------------------------- Quantity discount
def quantity_discount(annual_demand, ordering_cost, tiers, holding_rate=None, holding_cost=None):
    """tiers: list of {"min_qty": int, "unit_price": float}. All-units discount.
    Total cost = D*C + (D/Q)S + (Q/2)H, purchase cost included."""
    D, S = annual_demand, ordering_cost
    tiers = sorted(tiers, key=lambda t: t["min_qty"])
    mins = [t["min_qty"] for t in tiers] + [math.inf]
    rows = []
    for k, t in enumerate(tiers):
        C = t["unit_price"]
        H = holding_rate * C if holding_rate is not None else holding_cost
        q_star = math.sqrt(2 * D * S / H)
        upper = mins[k + 1]
        rng = f"{t['min_qty']:,}-{upper - 1:,}" if upper != math.inf else f"{t['min_qty']:,}+"
        if q_star >= upper - 1e-9:
            rows.append({"Price tier": rng, "Unit price ($)": C, "H ($/unit/yr)": _r(H, 4),
                         "EOQ at this price": _r(q_star), "Order qty used": None,
                         "Purchase cost ($)": None, "Ordering cost ($)": None, "Holding cost ($)": None,
                         "Total annual cost ($)": None,
                         "Status": "Infeasible: EOQ qualifies for a lower price"})
            continue
        Q = max(q_star, t["min_qty"])
        pc, oc, hc = D * C, D / Q * S, Q / 2 * H
        rows.append({"Price tier": rng, "Unit price ($)": C, "H ($/unit/yr)": _r(H, 4),
                     "EOQ at this price": _r(q_star), "Order qty used": _r(Q),
                     "Purchase cost ($)": _r(pc), "Ordering cost ($)": _r(oc), "Holding cost ($)": _r(hc),
                     "Total annual cost ($)": _r(pc + oc + hc),
                     "Status": "EOQ feasible" if Q == q_star else "EOQ below tier minimum: rounded up to break point"})
    feasible = [r for r in rows if r["Total annual cost ($)"] is not None]
    best = min(feasible, key=lambda r: r["Total annual cost ($)"])
    return {
        "tool": "Quantity Discount (all-units)",
        "inputs": {"Annual demand D (units/yr)": D, "Ordering cost S ($/order)": S,
                   "Holding cost": f"{holding_rate:.0%} of unit price" if holding_rate is not None
                   else f"${holding_cost}/unit/yr",
                   "Price tiers": [f"{t['min_qty']:,}+ units: ${t['unit_price']}" for t in tiers]},
        "conversions": [],
        "formula": "For each tier: H = i x C; Q* = sqrt(2DS/H). If Q* is below the tier minimum, use the "
                   "minimum; if Q* qualifies for a lower price, the tier is infeasible. "
                   "Total cost = DC + (D/Q)S + (Q/2)H. Choose the lowest total cost.",
        "candidates": rows,
        "results": {"Best order quantity (units)": best["Order qty used"], "Unit price ($)": best["Unit price ($)"],
                    "Annual purchase cost ($)": best["Purchase cost ($)"],
                    "Annual ordering cost ($)": best["Ordering cost ($)"],
                    "Annual holding cost ($)": best["Holding cost ($)"],
                    "Total annual cost ($)": best["Total annual cost ($)"]},
    }


# ---------------------------------------------------------------- Safety stock / ROP
def safety_stock_rop(avg_demand, demand_sd, demand_period, lead_time, lead_time_unit,
                     service_level, lead_time_sd=0.0):
    """Converts lead time into demand periods, then
    sigma_LT = sqrt(L*sd_d^2 + d^2*sd_L^2), SS = z*sigma_LT, ROP = d*L + SS."""
    conversions = []
    factor = DAYS_PER[lead_time_unit] / DAYS_PER[demand_period]
    L, sL = lead_time * factor, lead_time_sd * factor
    if lead_time_unit != demand_period:
        conversions.append(f"Lead time converted to {demand_period}s: {lead_time} {lead_time_unit}(s) x "
                           f"{factor:g} = {L:g} {demand_period}s; lead-time SD {lead_time_sd} x {factor:g} "
                           f"= {sL:g} {demand_period}s (7-day weeks)")
    z = norm.ppf(service_level)
    sigma = math.sqrt(L * demand_sd ** 2 + avg_demand ** 2 * sL ** 2)
    ss = z * sigma
    return {
        "tool": "Safety Stock & Reorder Point",
        "inputs": {f"Average demand d (units/{demand_period})": avg_demand,
                   f"Demand SD (units/{demand_period})": demand_sd,
                   f"Lead time L ({demand_period}s)": _r(L, 4), f"Lead-time SD ({demand_period}s)": _r(sL, 4),
                   "Cycle service level": f"{service_level:.1%}"},
        "conversions": conversions,
        "formula": "z = NORM.S.INV(service level); sigma_LT = sqrt(L x sd_d^2 + d^2 x sd_L^2); "
                   "SS = z x sigma_LT; ROP = d x L + SS",
        "results": {"z": _r(z, 4), "Demand during lead time d x L (units)": _r(avg_demand * L),
                    "Std dev of demand during lead time (units)": _r(sigma),
                    "Safety stock (units)": _r(ss), "Reorder point (units)": _r(avg_demand * L + ss)},
    }


# ---------------------------------------------------------------- Fixed-period review
def fixed_period_review(avg_demand, demand_sd, demand_period, review_period, lead_time, time_unit,
                        service_level, on_hand, on_order=0.0, backorders=0.0):
    factor = DAYS_PER[time_unit] / DAYS_PER[demand_period]
    T, L = review_period * factor, lead_time * factor
    conversions = [] if time_unit == demand_period else [
        f"Review period and lead time converted to {demand_period}s: T = {T:g}, L = {L:g}"]
    z = norm.ppf(service_level)
    sigma = demand_sd * math.sqrt(T + L)
    target = avg_demand * (T + L) + z * sigma
    ip = on_hand + on_order - backorders
    return {
        "tool": "Fixed-Period Review",
        "inputs": {f"Average demand (units/{demand_period})": avg_demand, f"Demand SD (units/{demand_period})": demand_sd,
                   f"Review period T ({demand_period}s)": _r(T, 4), f"Lead time L ({demand_period}s)": _r(L, 4),
                   "Service level": f"{service_level:.1%}", "On hand": on_hand, "On order": on_order,
                   "Backorders": backorders},
        "conversions": conversions,
        "formula": "Target = d(T+L) + z x sd_d x sqrt(T+L); Inventory position = on hand + on order - backorders; "
                   "Order quantity = Target - Inventory position",
        "results": {"z": _r(z, 4), "Safety stock (units)": _r(z * sigma), "Target inventory level (units)": _r(target),
                    "Inventory position (units)": _r(ip), "Order quantity (units)": _r(max(target - ip, 0))},
    }


# ---------------------------------------------------------------- Newsvendor
def newsvendor(selling_price, unit_cost, salvage_value, mean_demand, demand_sd, shortage_penalty=0.0):
    cu = selling_price - unit_cost + shortage_penalty
    co = unit_cost - salvage_value
    cr = cu / (cu + co)
    z = norm.ppf(cr)
    return {
        "tool": "Newsvendor",
        "inputs": {"Selling price ($)": selling_price, "Unit cost ($)": unit_cost, "Salvage value ($)": salvage_value,
                   "Shortage penalty ($)": shortage_penalty, "Mean demand": mean_demand, "Demand SD": demand_sd},
        "conversions": [],
        "formula": "Cu = price - cost + penalty; Co = cost - salvage; CR = Cu/(Cu+Co); z = NORM.S.INV(CR); Q* = mu + z x sigma",
        "results": {"Underage cost Cu ($)": _r(cu), "Overage cost Co ($)": _r(co), "Critical ratio": _r(cr, 4),
                    "z": _r(z, 4), "Optimal order quantity Q* (units)": _r(mean_demand + z * demand_sd)},
    }


# ---------------------------------------------------------------- Multicriteria classification
SCORE = {"A": 4, "B": 3, "C": 2, "D": 1}


def _pareto_class(cum_pct):
    if cum_pct <= 60:
        return "A"
    if cum_pct <= 80:
        return "B"
    if cum_pct <= 90:
        return "C"
    return "D"


def _final_class(score):
    if score >= 3.25:
        return "A"
    if score >= 2.50:
        return "B"
    if score >= 1.75:
        return "C"
    return "D"


def classify_inventory(df, sales_col="Sales ($)", hits_col="Hits"):
    """BAN 517 multicriteria method: separate Pareto classes on Sales and Hits (cumulative % of each
    criterion's total, unrounded; A<=60, B<=80, C<=90, D>90), scores A4 B3 C2 D1,
    combined = 0.65 Sales + 0.35 Hits (2 dp), final A>=3.25, B>=2.50, C>=1.75, else D."""
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    for col in (sales_col, hits_col):
        out[col] = pd.to_numeric(out[col].astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    out = out.dropna(subset=[sales_col, hits_col])
    for col, tag in ((sales_col, "Sales"), (hits_col, "Hits")):
        order = out[col].sort_values(ascending=False, kind="mergesort").index
        cum = out.loc[order, col].cumsum() / out[col].sum() * 100
        out[f"{tag} Cum %"] = cum.round(2)
        out[f"{tag} Class"] = cum.apply(_pareto_class)
        out[f"{tag} Score"] = out[f"{tag} Class"].map(SCORE)
    out["Combined Score"] = (0.65 * out["Sales Score"] + 0.35 * out["Hits Score"]).round(2)
    out["Final Class"] = out["Combined Score"].apply(_final_class)
    cols = ["Item Number", "Item Name", sales_col, "Sales Cum %", "Sales Class", "Sales Score", hits_col,
            "Hits Cum %", "Hits Class", "Hits Score", "Combined Score", "Final Class"]
    out = out[[c for c in cols if c in out.columns]].sort_values(sales_col, ascending=False)
    summary = (out.groupby("Final Class").agg(**{"Number of Items": (sales_col, "size"),
                                                  "Total Sales ($)": (sales_col, "sum")})
               .reindex(list("ABCD"), fill_value=0))
    summary["% of Total Sales"] = (summary["Total Sales ($)"] / summary["Total Sales ($)"].sum() * 100).round(2)
    total = pd.DataFrame({"Number of Items": [int(summary["Number of Items"].sum())],
                          "Total Sales ($)": [summary["Total Sales ($)"].sum()],
                          "% of Total Sales": [100.0]}, index=["Total"])
    summary = pd.concat([summary, total])
    summary.index.name = "Final Class"
    return out.reset_index(drop=True), summary
