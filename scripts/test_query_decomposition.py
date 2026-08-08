"""
scripts/test_query_decomposition.py
======================================
Verification test script for LLM-based Query Analysis & Query Decomposition.
"""

import sys
import os
import json

# Add VQA root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.query_refiner import get_query_refiner
from src.router import get_router

TEST_QUERIES = [
    {
        "query": "Người đàn ông mặc áo đỏ đang đưa chai nước cho ai?",
        "hints": []
    },
    {
        "query": "Người đàn ông mặc áo đen đưa chai nước cho người phụ nữ rồi đi vào cửa hàng ở đâu?",
        "hints": []
    },
    {
        "query": "Biển báo ven đường có viết dòng chữ gì?",
        "hints": []
    },
    {
        "query": "What does he do before entering the store?",
        "hints": ["A man wearing a black shirt enters a store and talks to a woman."]
    }
]

def main():
    print("=== TESTING LLM QUERY REFINER & DECOMPOSITION ===")
    refiner = get_query_refiner()
    router = get_router()

    for idx, t in enumerate(TEST_QUERIES, 1):
        q = t["query"]
        hints = t["hints"]
        print(f"\n--- Test #{idx} ---")
        print(f"Original Query: {q}")
        if hints:
            print(f"Context Hints: {hints}")

        refined = refiner.refine(q, context_hints=hints)
        routing = router.route(q, refined_query=refined)

        print("\n[Query Analysis]")
        print(f"  Canonical Question: {refined.analysis.canonical_question}")
        print(f"  Original Lang:     {refined.analysis.original_language}")
        print(f"  Intent:            {refined.analysis.intent}")
        print(f"  Modalities:        {refined.analysis.modalities}")
        print(f"  Keywords:          {refined.analysis.keywords}")
        print(f"  Entities:          {refined.analysis.entities}")
        print(f"  Attributes:        {refined.analysis.attributes}")

        print("\n[Query Decomposition]")
        print(f"  Needs Decomposition: {refined.decomposition.needs_decomposition}")
        print(f"  Sub-queries:         {refined.decomposition.sub_queries}")
        print(f"  Relations:           {refined.decomposition.relations}")

        print("\n[Gate A Routing (Heuristic)]")
        print(f"  Decision Tier: {routing.tier}")
        print(f"  Score:         {routing.complexity_score}")
        print(f"  Reason:        {routing.reason}")

if __name__ == "__main__":
    main()
