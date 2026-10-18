import json

import pytest
from transformers import AutoTokenizer

from src.training.data import tokenize_example

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def pair():
    with open("data/pairs/train.jsonl", encoding="utf-8") as f:
        return json.loads(f.readline())


def test_lengths_match(tok, pair):
    ex = tokenize_example(pair, tok, max_len=2048)
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


def test_prompt_is_masked_and_answer_is_not(tok, pair):
    ex = tokenize_example(pair, tok, max_len=2048)
    labels = ex["labels"]
    first = next(i for i, x in enumerate(labels) if x != -100)
    assert first > 0
    assert all(x == -100 for x in labels[:first])
    assert all(x != -100 for x in labels[first:])


def test_unmasked_text_is_the_target(tok, pair):
    ex = tokenize_example(pair, tok, max_len=2048)
    kept = [x for x in ex["labels"] if x != -100]
    text = tok.decode(kept)
    assert text.startswith(pair["target"])
    assert "<|im_end|>" in text
    assert "You extract structured data" not in text


def test_labels_equal_input_ids_where_unmasked(tok, pair):
    ex = tokenize_example(pair, tok, max_len=2048)
    for i, lab in zip(ex["input_ids"], ex["labels"]):
        assert lab == -100 or lab == i


def test_too_long_raises(tok, pair):
    with pytest.raises(ValueError):
        tokenize_example(pair, tok, max_len=50)