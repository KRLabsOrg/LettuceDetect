#!/usr/bin/env python3
"""Test script for hallucination tag handling in translate.py.

This script tests the find_hallucination_tags and put_hallucination_tags
functions to ensure they correctly handle HAL tags in the text.

To run:
    python scripts/translate/test_hal_tags.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lettucedetect.datasets.hallucination_dataset import HallucinationSample
from scripts.translate.translate import find_hallucination_tags, put_hallucination_tags


def test_basic_functions():
    """Test basic functionality of hallucination tag handling."""
    # Test 1: Basic tag removal
    print("Test 1: Basic tag removal")
    text = "This is a <HAL>hallucinated</HAL> text."
    labels = [{"start": 10, "end": 22, "label": "Wrong info"}]

    spans, cleaned_text = find_hallucination_tags(text, labels, 0)

    print(f"Original text: {text}")
    print(f"Cleaned text:  {cleaned_text}")
    print(f"Found spans:   {spans}")
    print("Expected:      [(10, 22, 'Wrong info')]")
    print(
        f"Correct:       {cleaned_text == 'This is a hallucinated text.' and len(spans) == 1 and spans[0][0] == 10 and spans[0][1] == 22}"
    )
    print()

    # Test 2: Multiple tags
    print("Test 2: Multiple tags")
    text = "<HAL>First</HAL> and <HAL>second</HAL> hallucination."
    labels = [
        {"start": 0, "end": 5, "label": "First error"},
        {"start": 11, "end": 17, "label": "Second error"},
    ]

    spans, cleaned_text = find_hallucination_tags(text, labels, 0)

    print(f"Original text: {text}")
    print(f"Cleaned text:  {cleaned_text}")
    print(f"Found spans:   {spans}")
    print("Expected:      [(0, 5, 'First error'), (11, 17, 'Second error')]")
    print(f"Correct:       {cleaned_text == 'First and second hallucination.' and len(spans) == 2}")
    print()

    # Test 3: End-to-end test (add tags then find them)
    print("Test 3: End-to-end test")

    # Create a sample with corrected spans that match the text length
    original_answer = "AI is an advanced technology that can do amazing things."
    sample = HallucinationSample(
        prompt="What can you tell me about AI?",
        answer=original_answer,
        labels=[
            {"start": 18, "end": 48, "label": "Hallucination"},
            {"start": 49, "end": 56, "label": "Hallucination"},
        ],
        split="test",
        task_type="qa",
        dataset="test_dataset",
        language="en",
    )

    tagged_answer, labels = put_hallucination_tags(sample, sample.answer)

    print(f"Original:      {sample.answer}")
    print(f"Tagged:        {tagged_answer}")

    spans, cleaned_text = find_hallucination_tags(tagged_answer, labels, 0)

    print(f"Cleaned:       {cleaned_text}")
    print(f"Found spans:   {spans}")

    # Verify that the spans match the original hallucination spans in the text
    first_span_correct = False
    second_span_correct = False
    if len(spans) >= 1:
        span_text = cleaned_text[spans[0][0] : spans[0][1]]
        expected_text = original_answer[sample.labels[0]["start"] : sample.labels[0]["end"]]
        first_span_correct = span_text == expected_text
        print(f"First span: '{span_text}' matches '{expected_text}': {first_span_correct}")

    if len(spans) >= 2:
        span_text = cleaned_text[spans[1][0] : spans[1][1]]
        expected_text = original_answer[sample.labels[1]["start"] : sample.labels[1]["end"]]
        second_span_correct = span_text == expected_text
        print(f"Second span: '{span_text}' matches '{expected_text}': {second_span_correct}")

    print(
        f"Round-trip success: {cleaned_text == sample.answer and len(spans) == len(labels) and first_span_correct and (len(spans) < 2 or second_span_correct)}"
    )


def test_edge_cases():
    """Test edge cases for hallucination tag handling."""
    # Test 1: Consecutive tags
    print("\nTest Edge Case 1: Consecutive tags")
    text = "<HAL>First</HAL><HAL>Second</HAL> text."
    labels = [
        {"start": 0, "end": 5, "label": "First error"},
        {"start": 5, "end": 11, "label": "Second error"},
    ]

    spans, cleaned_text = find_hallucination_tags(text, labels, 0)

    print(f"Original:    {text}")
    print(f"Cleaned:     {cleaned_text}")
    print(f"Found spans: {spans}")
    print(f"Correct:     {cleaned_text == 'FirstSecond text.' and len(spans) == 2}")
    print()

    # Test 2: Unmatched tags
    print("Test Edge Case 2: Unmatched tags")
    text = "This <HAL>has an opening tag but no closing tag."
    labels = [{"start": 5, "end": 40, "label": "Incomplete"}]

    spans, cleaned_text = find_hallucination_tags(text, labels, 0)

    print(f"Original:    {text}")
    print(f"Cleaned:     {cleaned_text}")
    print(f"Found spans: {spans}")
    print(
        f"Correct:     {len(spans) == 0 and 'has an opening tag but no closing tag' in cleaned_text}"
    )
    print()

    # Test 3: Tags in the middle of words
    print("Test Edge Case 3: Tags in the middle of words")
    text = "Un<HAL>detect</HAL>able hallucinations."
    labels = [{"start": 2, "end": 8, "label": "Mid-word error"}]

    spans, cleaned_text = find_hallucination_tags(text, labels, 0)

    print(f"Original:    {text}")
    print(f"Cleaned:     {cleaned_text}")
    print(f"Found spans: {spans}")
    print(f"Correct:     {cleaned_text == 'Undetectable hallucinations.' and len(spans) == 1}")


def test_position_preservation():
    """Test that hallucination span positions are preserved correctly after tag removal."""
    print("\nTest: Position Preservation")

    text = "This is a <HAL>hallucinated</HAL> text."
    labels = [{"start": 10, "end": 22, "label": "Wrong info"}]

    spans, cleaned_text = find_hallucination_tags(text, labels, 0)

    if len(spans) > 0:
        span_start, span_end, _ = spans[0]
        span_text = cleaned_text[span_start:span_end]
        expected_position_text = "hallucinated"

        print(f"Original text: {text}")
        print(f"Cleaned text:  {cleaned_text}")
        print(f"Span position: {span_start}-{span_end}")
        print(f"Span text:     '{span_text}'")
        print(f"Expected text: '{expected_position_text}'")
        print(f"Position preserved correctly: {span_text == expected_position_text}")

    print("\nMultiple spans test:")

    text = "This <HAL>has</HAL> multiple <HAL>hallucinated</HAL> spans in it."
    labels = [
        {"start": 5, "end": 8, "label": "Error 1"},
        {"start": 18, "end": 30, "label": "Error 2"},
    ]

    spans, cleaned_text = find_hallucination_tags(text, labels, 0)

    if len(spans) >= 2:
        span1_text = cleaned_text[spans[0][0] : spans[0][1]]
        expected1 = "has"
        print(f"First span: '{span1_text}' at position {spans[0][0]}-{spans[0][1]}")
        print(f"Expected:   '{expected1}' - Correct: {span1_text == expected1}")

        span2_text = cleaned_text[spans[1][0] : spans[1][1]]
        expected2 = "hallucinated"
        print(f"Second span: '{span2_text}' at position {spans[1][0]}-{spans[1][1]}")
        print(f"Expected:    '{expected2}' - Correct: {span2_text == expected2}")

        # Verify content between spans
        between_text = cleaned_text[spans[0][1] : spans[1][0]]
        expected_between = " multiple "
        print(f"Text between spans: '{between_text}'")
        print(
            f"Expected between:   '{expected_between}' - Correct: {between_text == expected_between}"
        )


if __name__ == "__main__":
    print("Testing hallucination tag handling functions...")
    print("=" * 60)

    # Run the tests
    test_basic_functions()
    test_edge_cases()
    test_position_preservation()

    print("\nAll tests completed!")
