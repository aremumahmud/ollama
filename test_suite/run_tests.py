import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import get_collection
from ask import answer_question, retrieve

TABLE_IMAGE_QUESTIONS = [
    "What does rule R1 say about age and student status for buying a computer?",
    "What is the reference citation for the data mining textbook mentioned at the end of the document?",
    "What are the layers described in the OLAM engine architecture diagram?",
    "In the decision tree diagram, what are the possible values for age?",
    "What do the fuzzy membership functions for income represent, and what are their labels?",
]

UNRELATED_QUESTIONS = [
    "What is Target Marketing in the context of data mining?",
    "What is the Agglomerative Approach in clustering?",
    "How do you define a rule-based hierarchy in DMQL, such as profit_margin_hierarchy?",
    "What are the theoretical foundations of data mining mentioned in Lesson Sixteen?",
    "What does Class/Concept refer to in data mining?",
]


def run_case(question: str, collection):
    documents, metadatas, distances = retrieve(question, collection)
    top_type = metadatas[0]["content_type"] if metadatas else "none"
    top_distance = distances[0] if distances else None

    answer = answer_question(question, collection)
    cited_table_or_image = "Table" in answer or "Image" in answer

    return {
        "question": question,
        "top_retrieved_type": top_type,
        "top_distance": round(top_distance, 4) if top_distance is not None else None,
        "cited_table_or_image": cited_table_or_image,
        "answer": answer,
    }


def print_case(i, result):
    print(f"[{i}] {result['question']}")
    print(f"    top retrieved chunk type: {result['top_retrieved_type']} (distance {result['top_distance']})")
    print(f"    cites a Table/Image in final answer: {result['cited_table_or_image']}")
    print(f"    answer:\n{result['answer']}")
    print("-" * 80)


def main():
    collection = get_collection(persist_dir="../chroma_db")

    print("=" * 80)
    print("PART 1: Table/Image-related questions (expect table/image citations)")
    print("=" * 80)
    for i, q in enumerate(TABLE_IMAGE_QUESTIONS, start=1):
        print_case(i, run_case(q, collection))

    print("=" * 80)
    print("PART 2: Table/Image-unrelated questions (expect text-only citations, no table/image)")
    print("=" * 80)
    for i, q in enumerate(UNRELATED_QUESTIONS, start=1):
        print_case(i, run_case(q, collection))


if __name__ == "__main__":
    main()
