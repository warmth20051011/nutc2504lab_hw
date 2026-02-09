import csv
from hybrid_retrieve import hybrid_retrieve
from rag_answer import rag_answer

QUESTIONS_CSV = "questions.csv"

def main():
    with open(QUESTIONS_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for i, r in enumerate(reader, start=1):
            question = r["題目"]

            print(f"\nQ{i}: {question}")

            contexts = hybrid_retrieve(
                query=question,
                top_k=5,   
                top_n=3
            )

            print(f"Retrieved {len(contexts)} chunks")

            answer = rag_answer(question, contexts)

            print("Answer:")
            print(answer)


if __name__ == "__main__":
    main()

