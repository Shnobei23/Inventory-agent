# Inventory Decision Agent — BAN 517 Project

## Goal
Extend the existing Streamlit Inventory Decision Agent with multicriteria inventory classification, and make sure every existing tool is analytically correct.

## Must keep working
EOQ, Quantity Discount, Safety Stock & Reorder Point, Fixed-Period Review, Newsvendor, and the existing CSV uploader (do NOT add a second uploader; reuse the uploaded dataframe).

## Rules
- All calculations are done in deterministic Python/pandas functions. The LLM never computes numbers itself; it only extracts inputs, calls functions, and explains results.
- Every result must display the extracted inputs (with units) used in the calculation.
- Change one thing at a time. Before editing, state which files/functions change and why. After editing, explain any formula change.
- Do not change a formula unless asked; if you think one is wrong, flag it and explain.

## Multicriteria classification method (authoritative — follow exactly)
Data columns: Item Number, Item Name, UoM, Sales ($), Hits. 205 items; Item Number is the unique key (some Item Names repeat — never group or dedupe by name). Strip `$`, commas, and spaces before converting Sales ($) and Hits to numbers. UoM is display-only.

1. Sales classification: sort by Sales descending; cumulative % = running sum / total Sales × 100. Class A if cum% ≤ 60, B if ≤ 80, C if ≤ 90, else D. Use unrounded cumulative %.
2. Hits classification: same procedure, sorted independently by Hits descending.
3. Scores: A=4, B=3, C=2, D=1.
4. Combined Score = 0.65 × Sales Score + 0.35 × Hits Score, rounded to 2 decimals.
5. Final Class: A if score ≥ 3.25; B if ≥ 2.50; C if ≥ 1.75; else D.
6. Output table columns: Item Number, Item Name, Sales ($), Sales Cum %, Sales Class, Sales Score, Hits, Hits Cum %, Hits Class, Hits Score, Combined Score, Final Class.
7. Summary table: number of items and total Sales ($) per Final Class (A, B, C, D), plus % of total Sales.

## Expected results for ProjectDataV2_UTF8.csv (regression check)
- Sales classes: A 12, B 28, C 23, D 142. Hits classes: A 18, B 40, C 30, D 117.
- Final: A 22 items / $54,788,937; B 18 / $10,450,897; C 23 / $8,173,036; D 142 / $8,633,280. Total 205 / $82,046,150.

## Correct tool formulas (check existing code against these)
- EOQ: Q* = sqrt(2DS/H); D annual; H = i × C if holding cost given as a rate.
- Quantity discount: per tier, H = i × price (if rate); compute Q*; if Q* < tier minimum, use the minimum; if Q* ≥ next tier minimum, tier is not a candidate. Total cost = D×price + (D/Q)S + (Q/2)H — purchase cost included. Choose lowest total cost.
- Safety stock/ROP: z = norm.ppf(cycle service level); σ_LT = sqrt(L·σd² + d²·σL²); SS = z·σ_LT; ROP = d·L + SS. Demand and lead-time units must match.
- Fixed-period review: target = d(T+L) + z·σd·sqrt(T+L); order = target − (on hand + on order − backorders).
- Newsvendor: Cu = price − cost (+ shortage penalty), Co = cost − salvage; CR = Cu/(Cu+Co); Q* = μ + z·σ with z = norm.ppf(CR).
