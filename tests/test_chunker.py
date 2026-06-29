"""Chunker unit tests."""

import pytest

from ragstore.chunker import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_single_chunk():
    assert chunk_text("hello world", chunk_size=500) == ["hello world"]


def test_splits_with_overlap():
    words = " ".join(str(i) for i in range(10))
    chunks = chunk_text(words, chunk_size=4, chunk_overlap=1)
    # step = 3 → starts at 0,3,6; the third chunk reaches the end and stops.
    assert chunks == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_no_overlap():
    words = " ".join(str(i) for i in range(6))
    chunks = chunk_text(words, chunk_size=3, chunk_overlap=0)
    assert chunks == ["0 1 2", "3 4 5"]


def test_invalid_params_fail_loud():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=3, chunk_overlap=3)
