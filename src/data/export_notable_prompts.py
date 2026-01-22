"""Export notable_3way.csv rows into JSONL prompts for activation collection."""

from pathlib import Path
import json
import pandas as pd


def main():
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / "src" / "data" / "datasets" / "notable_3way.csv"
    out_path = repo_root / "src" / "data" / "prompts" / "notable_3way.jsonl"

    df = pd.read_csv(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dates to numerical labels using (year - 1900) / 100
    labels = (pd.to_datetime(df["correct_date"]).dt.year - 1900) / 100

    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            record = {
                "text": row["context"],
                "label": round(labels.iloc[idx], 2),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df)} prompts to {out_path}")


if __name__ == "__main__":
    main()
