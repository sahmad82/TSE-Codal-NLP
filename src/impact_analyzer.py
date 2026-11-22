from typing import Dict
import pandas as pd
from scipy import stats


def analyze_sentiment_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    ارزیابی آماری بازده سهام پس از انتشار اطلاعیه‌های کدال به تفکیک لحن (Bullish, Neutral, Bearish)
    به همراه آزمون فرض Welch's t-test در مقایسه با اطلاعیه‌های خنثی.
    """
    rows = []
    neutral_df = df[df["sentiment_label"] == "Neutral"]

    for label in ("Bullish", "Neutral", "Bearish"):
        sub = df[df["sentiment_label"] == label]
        if sub.empty:
            continue

        row: Dict[str, object] = {
            "Sentiment": label,
            "Disclosures": int(len(sub)),
            "Avg Score": round(float(sub["sentiment_score"].mean()), 3),
        }

        for h in (1, 3, 5):
            col = f"ret_{h}d"
            vals = sub[col].dropna()
            neut_vals = neutral_df[col].dropna()

            row[f"Avg {h}D Ret (%)"] = round(float(vals.mean() * 100.0), 2)
            row[f"Win Rate {h}D (%)"] = round(float((vals > 0).mean() * 100.0), 1)

            if label != "Neutral" and len(vals) >= 3 and len(neut_vals) >= 3:
                t_stat, p_val = stats.ttest_ind(vals, neut_vals, equal_var=False)
                row[f"t-stat ({h}D)"] = round(float(t_stat), 2)
                row[f"p-val ({h}D)"] = round(float(p_val), 4)
            else:
                row[f"t-stat ({h}D)"] = 0.0
                row[f"p-val ({h}D)"] = 1.0

        rows.append(row)

    return pd.DataFrame(rows)


def extract_top_drivers(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """استخراج پرتکرارترین محرک‌های بنیادی (Drivers) شناسایی‌شده در اطلاعیه‌های کدال."""
    drivers_series = (
        df[df["matched_drivers"] != "None"]["matched_drivers"]
        .str.split(", ")
        .explode()
    )
    if drivers_series.empty:
        return pd.DataFrame(columns=["Driver Phrase", "Occurrences"])

    counts = drivers_series.value_counts().head(top_n).reset_index()
    counts.columns = ["Driver Phrase", "Occurrences"]
    return counts


if __name__ == "__main__":
    codal_df = pd.read_csv("data/codal_announcements.csv")
    impact_table = analyze_sentiment_impact(codal_df)
    print("\n=== Statistical Event Study: Codal Sentiment vs. Post-Disclosure Returns ===")
    print(impact_table.to_string(index=False))

    print("\n=== Top Extracted Fundamental Drivers ===")
    print(extract_top_drivers(codal_df).to_string(index=False))