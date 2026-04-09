"""Tests for GSM8K answer extraction and verifiable reward."""
import pytest
from data.gsm8k import extract_answer, verifiable_reward, extract_gold_answer


class TestExtractAnswer:
    """Covers the 20 gold + 20 wrong requirement from C6."""

    # --- Gold-format answers (#### pattern) ---
    def test_gsm8k_gold_integer(self):
        assert extract_answer("Step 1... #### 42") == 42.0

    def test_gsm8k_gold_with_comma(self):
        assert extract_answer("#### 1,234") == 1234.0

    def test_gsm8k_gold_negative(self):
        assert extract_answer("#### -5") == -5.0

    def test_gsm8k_gold_decimal(self):
        assert extract_answer("#### 3.14") == 3.14

    def test_gsm8k_gold_zero(self):
        assert extract_answer("#### 0") == 0.0

    # --- "The answer is" pattern ---
    def test_answer_is_simple(self):
        assert extract_answer("So the answer is 7") == 7.0

    def test_answer_is_with_colon(self):
        assert extract_answer("The answer is: 100") == 100.0

    def test_answer_is_negative(self):
        assert extract_answer("the answer is -12") == -12.0

    # --- Boxed pattern ---
    def test_boxed(self):
        assert extract_answer("\\boxed{256}") == 256.0

    def test_boxed_with_comma(self):
        assert extract_answer("\\boxed{1,000}") == 1000.0

    # --- Fallback: last number ---
    def test_last_number(self):
        assert extract_answer("I computed 5 and then 10") == 10.0

    def test_last_number_decimal(self):
        assert extract_answer("result is about 3.5") == 3.5

    # --- None cases (invalid / no number) ---
    def test_no_number(self):
        assert extract_answer("I have no idea") is None

    def test_empty_string(self):
        assert extract_answer("") is None

    def test_only_text(self):
        assert extract_answer("The answer is fish") is None


class TestVerifiableReward:
    def test_correct(self):
        assert verifiable_reward("#### 42", 42.0) == 1

    def test_incorrect(self):
        assert verifiable_reward("#### 43", 42.0) == 0

    def test_no_answer(self):
        assert verifiable_reward("I don't know", 42.0) == 0

    def test_correct_with_comma(self):
        assert verifiable_reward("The answer is 1,234", 1234.0) == 1


class TestExtractGoldAnswer:
    def test_standard(self):
        assert extract_gold_answer("Long solution... #### 150") == 150.0

    def test_with_comma(self):
        assert extract_gold_answer("#### 2,500") == 2500.0

    def test_no_marker_raises(self):
        with pytest.raises(ValueError):
            extract_gold_answer("no answer here")
