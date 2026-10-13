import json

file_path = "evaluation_results/answer_accuracy_results.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for result in data["evaluation"]["results"]:
   if result["answer_accuracy_percent"] < 95:
        print("\n" + "=" * 70)
        print(f"Q{result['question_number']}")
        print("=" * 70)

        print("QUESTION:")
        print(result["question"])

        print("\nGENERATED ANSWER:")
        print(result["generated_answer"])

        print("\nMATCHED CONCEPTS:")
        print(result["matched_concepts"])

        print("\nMISSING CONCEPTS:")
        print(result["missing_concepts"])