test_cases = [
    {
        "query": "refund after 20 days",
        "expected_keywords": ["no refunds after 14 days"]
    },
    {
        "query": "pricing plans",
        "expected_keywords": ["free", "basic", "pro", "enterprise"]
    },
    {
        "query": "failed payment refund",
        "expected_keywords": ["failed payments", "refund"]
    },
]

def evaluate(rag_pipeline):
    hits = 0

    for case in test_cases:
        print(f"\n--- Testing: {case['query']} ---")

        answer = rag_pipeline.generate_answer(case["query"]).lower()
        print("Answer:", answer)

        if any(k in answer for k in case["expected_keywords"]):
            print("✅ PASS")
            hits += 1
        else:
            print("❌ FAIL")

    print(f"\nFinal Accuracy: {hits}/{len(test_cases)}")