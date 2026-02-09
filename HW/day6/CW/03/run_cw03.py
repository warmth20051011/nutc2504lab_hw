import csv
from query_rewrite import rewrite_query
from retrieve_and_answer import retrieve, answer

QUESTIONS_CSV = "questions.csv"


def main():
    with open(QUESTIONS_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for i, r in enumerate(reader, start=1):
            question = r["題目"]   

            print(f"\nQ{i}: {question}")

            rq = rewrite_query(question)
            print(f" Rewrite: {rq}")

            contexts = retrieve(rq)
            print(f" Retrieved {len(contexts)} chunks")

            final_answer = answer(question, contexts)

            print("💡 Answer:")
            print(final_answer)


if __name__ == "__main__":
    main()

