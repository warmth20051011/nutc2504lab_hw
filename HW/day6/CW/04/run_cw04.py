import csv
from hybrid_retrieve import hybrid_retrieve
from rag_answer import answer_question

QUESTIONS_CSV = "questions.csv"


def main():
    with open(QUESTIONS_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for i, r in enumerate(reader, start=1):
            question = r.get("題目") or r.get("questions")

            print(f"\nQ{i}: {question}")

            contexts = hybrid_retrieve(question, top_k=5)
            print(f"Retrieved {len(contexts)} chunks")

            answer = answer_question(question, contexts)

            print("Answer:")
            print(answer)
            print("-" * 50)


if __name__ == "__main__":
    main()

