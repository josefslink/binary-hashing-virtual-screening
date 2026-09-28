#!/usr/bin/env python3
# re-derive every number quoted in docs/report/report.tex from the CSV it came from.
# labels below name the float in that report: T3 = Table 3, F2 = Figure 2, S5.2 = Section 5.2.
# exit non-zero on a mismatch.

import pandas as pd, numpy as np, json, sys
from pathlib import Path

M = Path("outputs/metrics")
fails, checks = [], 0

def mean(csv, col="auroc"):
    d = pd.read_csv(M / csv)
    if "target" in d:
        d = d[d["target"].astype(str).str.lower() != "mean"]  # a summary Mean row would double-count
    return d[col].mean()

def chk(label, claimed, actual, tol=5e-5):
    # 5e-5 is source precision; table cells that were rounded to 2-3 decimals pass 5e-3
    global checks; checks += 1
    ok = abs(claimed - actual) <= tol
    if not ok:
        fails.append(f"{label}: report={claimed} source={actual:.6f} diff={abs(claimed-actual):.6f}")
    print(f"{'ok ' if ok else 'FAIL'} {label:52s} report={claimed:<10} source={actual:.6f}")

# claimed numbers below are transcribed from the report. this catches CSV drift, not a
# digit edited in the .tex - keep the two in step by hand when a table changes.
# Table 3, LIT-PCBA half
S = "lit_pcba_results_cosine_20260824_232705.csv"
D = "drugclip_pcba_cosine_20260825_004540.csv"
for col, c in [("auroc",0.7253),("bedroc",0.1106),("ef_0.005",13.71),("ef_0.01",9.55),("ef_0.05",5.13)]:
    chk(f"T3 LIT-PCBA SPRINT {col}", c, mean(S,col), 5e-3)
for col, c in [("auroc",0.5832),("bedroc",0.0598),("ef_0.005",6.62),("ef_0.01",5.18),("ef_0.05",2.18)]:
    chk(f"T3 LIT-PCBA DrugCLIP {col}", c, mean(D,col), 5e-3)

# Table 3, DUD-E half
DD = "drugclip_dude_cosine_20260825_150207.csv"
SD = "sprint_dude_results_cosine_20260826_120525.csv"
for col, c in [("auroc",0.8071),("bedroc",0.4733),("ef_0.005",36.46),("ef_0.01",30.00),("ef_0.05",10.35)]:
    chk(f"T3 DUD-E DrugCLIP {col}", c, mean(DD,col), 5e-3)
for col, c in [("auroc",0.6759),("bedroc",0.0700),("ef_0.005",3.17),("ef_0.01",3.34),("ef_0.05",2.77)]:
    chk(f"T3 DUD-E SPRINT {col}", c, mean(SD,col), 5e-3)

# Table 4, the centering grid. values are AUROC; csv filename encodes metric+center.
GRID = {
 ("SPRINT","LIT-PCBA"): {"cosine":(S,0.7253),
   "none":("lit_pcba_results_hamming_20260824_233650.csv",0.5442),
   "per_type":("lit_pcba_results_hamming_per_type_20260825_130648.csv",0.6662),
   "per_type_median":("lit_pcba_results_hamming_per_type_median_20260826_235432.csv",0.5201),
   "global":("lit_pcba_results_hamming_global_20260824_235309.csv",0.4274),
   "global_median":("lit_pcba_results_hamming_global_median_20260826_235624.csv",0.3533)},
 ("SPRINT","DUD-E"): {"cosine":(SD,0.6759),
   "none":("sprint_dude_results_hamming_20260826_122228.csv",0.5285),
   "per_type":("sprint_dude_results_hamming_per_type_20260826_124015.csv",0.5535),
   "per_type_median":("sprint_dude_results_hamming_per_type_median_20260826_235721.csv",0.5947),
   "global":("sprint_dude_results_hamming_global_20260826_130116.csv",0.3989),
   "global_median":("sprint_dude_results_hamming_global_median_20260826_235817.csv",0.4201)},
 ("DrugCLIP","LIT-PCBA"): {"cosine":(D,0.5832),
   "none":("drugclip_pcba_hamming_20260825_004551.csv",0.5780),
   "per_type":("drugclip_pcba_hamming_per_type_20260825_125817.csv",0.5547),
   "per_type_median":("drugclip_pcba_hamming_per_type_median_20260826_174109.csv",0.5407),
   "global":("drugclip_pcba_hamming_global_20260825_004612.csv",0.6006),
   "global_median":("drugclip_pcba_hamming_global_median_20260826_174140.csv",0.5991)},
 ("DrugCLIP","DUD-E"): {"cosine":(DD,0.8071),
   "none":("drugclip_dude_hamming_20260825_150214.csv",0.7918),
   "per_type":("drugclip_dude_hamming_per_type_20260825_150221.csv",0.7885),
   "per_type_median":("drugclip_dude_hamming_per_type_median_20260826_174154.csv",0.7881),
   "global":("drugclip_dude_hamming_global_20260825_150228.csv",0.8261),
   "global_median":("drugclip_dude_hamming_global_median_20260826_174214.csv",0.8280)},
}
for (m,b), conds in GRID.items():
    for cond,(csv,claimed) in conds.items():
        chk(f"T4 {m}/{b}/{cond}", claimed, mean(csv), 5e-5)

# Table 5 and the Section 5.2 deltas. "lambda0" here is post-hoc binarization of
# frozen DrugCLIP, not the trained lam0p00 head in HEAD below.
chk("T5 none BEDROC", 0.4217, mean("drugclip_dude_hamming_20260825_150214.csv","bedroc"), 5e-4)
chk("T5 none EF0.5",  32.59,  mean("drugclip_dude_hamming_20260825_150214.csv","ef_0.005"), 5e-3)
chk("T5 global BEDROC",0.4461, mean("drugclip_dude_hamming_global_20260825_150228.csv","bedroc"), 5e-4)
chk("T5 global EF0.5", 33.60,  mean("drugclip_dude_hamming_global_20260825_150228.csv","ef_0.005"), 5e-3)
chk("S5.2 global vs cosine dAUROC", 0.0190,
    mean("drugclip_dude_hamming_global_20260825_150228.csv")-mean(DD), 5e-4)
chk("S5.2 global vs cosine dBEDROC", -0.0272,
    mean("drugclip_dude_hamming_global_20260825_150228.csv","bedroc")-mean(DD,"bedroc"), 5e-4)
chk("S5.2 global vs cosine dEF0.5", -2.86,
    mean("drugclip_dude_hamming_global_20260825_150228.csv","ef_0.005")-mean(DD,"ef_0.005"), 5e-3)
chk("S5.2 pcba global vs cosine dAUROC", 0.0174,
    mean("drugclip_pcba_hamming_global_20260825_004612.csv")-mean(D), 5e-4)
chk("S5.2 SPRINT pcba per_type EF0.5", 4.40,
    mean("lit_pcba_results_hamming_per_type_20260825_130648.csv","ef_0.005"), 5e-3)

# leakage from check_drughash_leakage.py. integers, so tol=0.
lk = json.loads((M / "drughash_leakage.json").read_text())
for bench, claimed in [("dude", (22805, 583, 90, 102)), ("lit_pcba", (9954, 36, 10, 15))]:
    d = lk["benchmarks"][bench]
    # targets with >=1 leaked active, not a leak count
    affected = sum(1 for v in d["per_target"].values() if v["leaked"] > 0)
    chk(f"T7 {bench} actives", claimed[0], d["actives"], 0)
    chk(f"T7 {bench} leaked", claimed[1], d["leaked"], 0)
    chk(f"T7 {bench} targets affected", claimed[2], affected, 0)
    chk(f"T7 {bench} targets total", claimed[3], len(d["per_target"]), 0)

# Figure 2 and Section 5.5, the trained hash head. tag is the sweep filename;
# none/global is binarize-center.
# lam0p00 is trained lambda=0 (L_c only), not the untrained post-hoc 0.7918 in the grid above.
H = M / "drugclip_dude_heads"
HEAD = {
    ("lam0p00", "none"): (0.7853, 0.4080, 31.45),
    ("lam1p00", "none"): (0.7937, 0.4369, 33.33),
    ("lam3p00", "none"): (0.8025, 0.4655, 35.72),
    ("lam10p00", "none"): (0.7995, 0.4680, 35.63),
    ("lam100p00", "none"): (0.7875, 0.4540, 33.90),
    ("lam3p00", "global"): (0.8333, 0.4774, 36.34),
}
for (tag, cond), (a, b, e) in HEAD.items():
    csv = f"drugclip_dude_heads/head_{tag}_{cond}.csv"
    chk(f"F2 {tag}/{cond} auroc", a, mean(csv), 5e-5)
    chk(f"F2 {tag}/{cond} bedroc", b, mean(csv, "bedroc"), 5e-5)
    chk(f"F2 {tag}/{cond} ef0.5", e, mean(csv, "ef_0.005"), 5e-3)

# Table 6, SPRINT under truncation with a seeded random tie-break. per_type
# centering, keeping the first k dimensions.
TR = "sprint_truncation/sprint_{b}_truncate_{k}bit.csv"
for b, lbl, rows in [
    ("pcba", "LIT-PCBA", [(1024,0.6606,4.15),(512,0.6527,3.47),(256,0.6483,3.89),(128,0.6417,2.81),(64,0.6460,2.70)]),
    ("dude", "DUD-E",    [(1024,0.5477,1.89),(512,0.5437,1.73),(256,0.5423,2.02),(128,0.5534,1.96),(64,0.5472,2.01)]),
]:
    for k, a, e in rows:
        csv = TR.format(b=b, k=k)
        chk(f"T6 {lbl} {k}bit auroc", a, mean(csv), 5e-5)
        chk(f"T6 {lbl} {k}bit ef0.5", e, mean(csv, "ef_0.005"), 5e-3)

# inner-val diagnostics, not screening AUROC
sweep = {r["lam"]: r for r in json.loads(Path("outputs/drughash/heads/sweep_summary.json").read_text())}
for lam, vc, gr, qq, bf, nc in [
    (0.0, 0.0368, 0.000, 0.8024, 0.121, 4013),
    (0.2, 0.0369, 0.023, 0.8287, 0.162, 3895),
    (1.0, 0.0370, 0.115, 0.8424, 0.227, 3826),
    (3.0, 0.0387, 0.340, 0.8491, 0.252, 3768),
    (10.0, 0.0613, 1.072, 0.8775, 0.302, 3669),
    (100.0, 0.2961, 1.927, 0.9196, 0.370, 3366),
]:
    r = sweep[lam]
    chk(f"S5.5 {lam} val_c", vc, r["val_c"], 5e-5)
    chk(f"S5.5 {lam} grad_ratio", gr, r["grad_ratio"], 5e-4)
    chk(f"S5.5 {lam} quant", qq, r["quant_quality"], 5e-5)
    chk(f"S5.5 {lam} flip", bf, r["bit_flip_fraction"], 5e-4)
    chk(f"S5.5 {lam} codes", nc, r["n_distinct_codes"], 0)  # integer count

print(f"\n{checks} checks, {len(fails)} failures")
for f in fails: print("  FAIL", f)
sys.exit(1 if fails else 0)
