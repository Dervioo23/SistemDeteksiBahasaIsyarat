"""
Tes gaya unit ringan untuk SequenceHandler dan ResponseEngine.
Dirancang agar cepat dan mudah dijalankan dari folder All_Testing.
"""

import time

from inference.sequence_handler import SequenceHandler
from inference.response_engine import ResponseEngine


def test_sequence_builds_word_and_completes() -> None:
    handler = SequenceHandler(
        max_sequence_length=10,
        letter_timeout=5.0,
        word_complete_timeout=0.5,
    )

    for letter in ["H", "A", "L", "O"]:
        added = handler.add_letter(letter)
        assert added, f"Letter {letter} should be accepted"
        # Tidur sebentar untuk mensimulasikan waktu antar huruf, tapi < letter_timeout
        time.sleep(0.1)

    # Tunggu cukup lama agar kata dianggap selesai
    time.sleep(handler.word_complete_timeout + 0.1)

    assert handler.is_word_complete(), "Word should be marked complete after timeout"
    word = handler.complete_word()
    assert word == "HALO", f"Expected word 'HALO', got '{word}'"
    assert handler.get_current_word() == "", "Sequence should reset after completion"


def test_sequence_rejects_fast_duplicates() -> None:
    handler = SequenceHandler(
        max_sequence_length=10,
        letter_timeout=5.0,
        word_complete_timeout=2.0,
    )

    assert handler.add_letter("A"), "First 'A' should be accepted"
    time.sleep(0.1)
    # Duplikat dalam 0.5 detik harus ditolak
    assert not handler.add_letter("A"), "Fast duplicate 'A' should be rejected"

    # Setelah cukup waktu berlalu, duplikat diperbolehkan
    time.sleep(0.6)
    assert handler.add_letter("A"), "Duplicate 'A' after delay should be accepted"


def test_response_engine_basic_behaviour() -> None:
    engine = ResponseEngine()

    # Kata sapaan harus dipetakan ke salah satu respons sapaan yang telah ditentukan
    resp = engine.process_word("halo")
    assert resp in engine.greeting_responses

    # Kata yang tidak diketahui harus kembali ke pola umum
    resp2 = engine.process_word("xyz")
    assert "Terdeteksi" in resp2

    # Pemetaan huruf harus mengembalikan string yang tidak kosong
    resp_letter = engine.process_letter("C")
    assert isinstance(resp_letter, str) and len(resp_letter) > 0

    # Kata yang dieja harus menggunakan templat kata yang dieja
    resp_spelled = engine.process_spelled_word("CAT")
    assert "Kata yang dieja" in resp_spelled


def run_all() -> None:
    test_sequence_builds_word_and_completes()
    test_sequence_rejects_fast_duplicates()
    test_response_engine_basic_behaviour()
    print("\n✅ test_unit_sequence_and_response.py: all tests passed")


if __name__ == "__main__":
    run_all()
