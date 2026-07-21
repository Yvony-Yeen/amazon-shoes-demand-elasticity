#!/usr/bin/env python3
"""
plot_elasticity_vs_price.py -- per-brand demand elasticity vs average price.

Reads the executed 04_evaluation_final_all logs for every *_lora bucket and its
non-LoRA counterpart (both txtimg and txt), plus each brand's average buy-box
price, and draws a 4-line chart:
    x = average buy-box price (USD, exp of the log price in the data)
    y = demand elasticity (theta = 0.6 -> multiplier 1.667)
    lines = {non-LoRA, LoRA} x {txtimg, txt}, points = brands sorted by price.

Outputs:
    data/_shared/elasticity_vs_price.png
    data/_shared/elasticity_vs_price.csv

Run:  python plot_elasticity_vs_price.py
"""
import json, re, glob, os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


def cells(nb):
    return json.load(open(nb))["cells"]


def _otext(c):
    s = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            s.append("".join(o.get("text", [])))
        elif "data" in o and "text/plain" in o.get("data", {}):
            s.append("".join(o["data"]["text/plain"]))
    return "\n".join(s)


def elast(nb):
    """Demand elasticity point estimate from a 04 log, or None."""
    if not os.path.exists(nb):
        return None
    t = "\n".join(_otext(c) for c in cells(nb) if c["cell_type"] == "code")
    m = re.search(r"Demand elasticity \(× 1/[^\)]*\):\s*([-\d.]+)", t)
    return float(m.group(1)) if m else None


def price_of(bucket):
    """Average buy-box price in USD (data stores natural-log price)."""
    for cand in [
        f"data/{bucket}/amzn_shoes_monthly_diffs_ffill_fixed_splits/all_products_clean.parquet",
        f"data/{bucket}_lora/amzn_shoes_monthly_diffs_ffill_fixed_splits/all_products_clean.parquet",
    ]:
        if os.path.exists(cand):
            df = pd.read_parquet(cand, columns=["BUYBOX_PRICE"])
            return float(np.exp(df["BUYBOX_PRICE"]).mean())
    return None


def logpath(bucket, lora, variant):
    base = f"data/{bucket}{'_lora' if lora else ''}/_run_logs/"
    return base + ("04_evaluation_final_all_txt.ipynb" if variant == "txt"
                   else "04_evaluation_final_all.ipynb")


def main():
    buckets = [os.path.basename(d).replace("_lora", "") for d in sorted(glob.glob("data/*_lora"))]
    rec = []
    for b in buckets:
        rec.append(dict(
            bucket=b, price=price_of(b),
            nl_ti=elast(logpath(b, False, "txtimg")), nl_tx=elast(logpath(b, False, "txt")),
            lo_ti=elast(logpath(b, True, "txtimg")),  lo_tx=elast(logpath(b, True, "txt")),
        ))
    df = pd.DataFrame(rec).dropna(subset=["price"]).sort_values("price").reset_index(drop=True)
    df.to_csv("data/_shared/elasticity_vs_price.csv", index=False)
    print(df.round(3).to_string())

    for variant in ("txtimg", "txt"):
        make_plot(df, variant)


def make_plot(df, variant):
    """One 2-line chart (non-LoRA vs LoRA) for a given modality."""
    nl_col, lo_col = ("nl_ti", "lo_ti") if variant == "txtimg" else ("nl_tx", "lo_tx")
    x = df["price"].values
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x, df[nl_col].values, marker="o", linestyle="-",  color="#1f77b4",
            label=f"non-LoRA ({variant})", linewidth=1.6, markersize=7, alpha=0.9)
    ax.plot(x, df[lo_col].values, marker="^", linestyle="--", color="#d62728",
            label=f"LoRA ({variant})", linewidth=1.6, markersize=7, alpha=0.9)
    ax.axhline(0, color="black", lw=1, ls=":")

    # number each point 1..N above the higher of the two markers (shared x-positions)
    ymax = df[[nl_col, lo_col]].max(axis=1).values
    for i, (xi, yi) in enumerate(zip(x, ymax), start=1):
        ax.annotate(str(i), (xi, yi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#333333")

    ax.set_xlabel("Average buy-box price (USD)", fontsize=12)
    ax.set_ylabel("Demand elasticity  (x 1/theta = 1.667)", fontsize=12)
    ax.set_title(f"Demand elasticity vs average price, per brand -- {variant} (theta=0.6)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, framealpha=0.9, loc="lower left")

    # index -> brand ($price) mapping, listed to the right of the plot
    mapping = "\n".join(f"{i:>2}. {b}  (${p:.0f})"
                        for i, (b, p) in enumerate(zip(df["bucket"], df["price"]), start=1))
    fig.subplots_adjust(right=0.72)
    fig.text(0.735, 0.5, mapping, va="center", ha="left", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#cccccc"))
    out = f"data/_shared/elasticity_vs_price_{variant}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()