#!/usr/bin/env python3
"""
scripts/test_cascade.py
========================
Test script to verify Bilingual Query Refinement, Gate A Router, and Gate B Confidence Evaluator.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(str(Path(__file__).parent.parent))

from src.query_refiner import get_query_refiner
from src.router import get_router
from src.confidence import evaluate_gate_b_confidence


def test_cascade():
    print("--- Testing Bilingual Query Refining & Cascade Routing ---\n")

    refiner = get_query_refiner()
    router = get_router()

    test_questions = [
        "Cho tôi biết chữ viết trên biển báo màu xanh là gì?",
        "Diễn viên chính đã phát biểu câu gì ở phút thứ 3?",
        "Cho tôi biết người đàn ông mặc áo màu đỏ đang làm gì bên cạnh chiếc xe màu vàng?",
        "What is the man wearing a blue hat holding in his hands?",
        "How many cars are parked in front of the building?"
    ]

    for q in test_questions:
        print(f"Original Question: '{q}'")
        refined = refiner.refine(q)
        print(f"  -> Lang: {refined.original_lang}")
        print(f"  -> Target Attr: {refined.target_attribute}")
        print(f"  -> Visual Prompt (VI): '{refined.visual_query_vi}'")
        print(f"  -> Visual Prompt (EN): '{refined.visual_query_en}'")
        print(f"  -> Keywords (VI): {refined.keywords_vi}")
        print(f"  -> Keywords (EN): {refined.keywords_en}")

        routing = router.route(q, refined_query=refined)
        print(f"  -> Gate A Route: {routing.tier} (Reason: {routing.reason})")

        # Mock candidates test for Gate B
        mock_candidates = [
            {"video_id": "video_001", "frame_id": 10, "score": 0.045, "source": "clip", "ocr_text": "STOP"},
            {"video_id": "video_001", "frame_id": 15, "score": 0.038, "source": "asr"},
            {"video_id": "video_002", "frame_id": 20, "score": 0.020, "source": "clip"}
        ]
        gate_b = evaluate_gate_b_confidence(q, mock_candidates, refined_query=refined)
        print(f"  -> Gate B Conf: {gate_b.confidence_score} (Escalate: {gate_b.should_escalate})\n")

    print("✅ Cascade Test Completed Successfully!")


if __name__ == "__main__":
    test_cascade()
