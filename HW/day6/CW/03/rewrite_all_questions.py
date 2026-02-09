import csv
from query_rewrite import rewrite_query

INPUT_CSV = "questions.csv"
OUTPUT_CSV = "Re_Write_questions.csv"


def main():
    rows = []

    with open(INPUT_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        print("CSV 欄位：", reader.fieldnames)

        for r in reader:
            q_id = r["題目_ID"]
            question = r["題目"]

            print(f"Rewrite Q{q_id}: {question}")

            rq = rewrite_query(question).strip()

            rows.append({
                "題目_ID": q_id,
                "題目": question,
                "Rewrite_Question": rq
            })

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["題目_ID", "題目", "Rewrite_Question"],
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ 已完成：{OUTPUT_CSV}")

if __name__ == "__main__":
    main()

