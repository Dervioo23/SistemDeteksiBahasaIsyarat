"""
Tes gaya unit ringan untuk fungsi normalisasi.
Menggunakan data sintetis dan menegaskan invarian dasar sehingga cepat dan deterministik.
"""

import numpy as np

from preprocessing.normalize import (
    normalize_landmarks_wrist_relative,
    normalize_landmarks_scale,
    normalize_landmarks_full,
)
from inference.utils import normalize_landmarks


def _make_dummy_single_frame() -> np.ndarray:
    """Buat sampel dummy kecil dengan kedua tangan ada.

    Bentuk: (1, 126) → 1 frame, 42 landmark × 3 koordinat.
    Pergelangan tangan kanan di indeks 0, pergelangan tangan kiri di indeks 63.
    """
    landmarks = np.zeros((1, 126), dtype=np.float32)

    # Pergelangan tangan kanan dan satu landmark tambahan
    landmarks[0, 0:3] = np.array([0.5, 0.5, 0.5], dtype=np.float32)      # pergelangan tangan
    landmarks[0, 3:6] = np.array([0.6, 0.4, 0.5], dtype=np.float32)      # titik lain

    # Pergelangan tangan kiri dan satu landmark tambahan
    landmarks[0, 63:66] = np.array([0.2, 0.3, 0.4], dtype=np.float32)    # pergelangan tangan
    landmarks[0, 66:69] = np.array([0.3, 0.5, 0.4], dtype=np.float32)    # titik lain

    return landmarks


def test_wrist_relative_centers_on_wrist() -> None:
    data = _make_dummy_single_frame()
    normalized = normalize_landmarks_wrist_relative(data)

    # Pergelangan tangan kanan harus menjadi titik asal
    assert np.allclose(normalized[0, 0:3], 0.0, atol=1e-6)
    # Pergelangan tangan kiri harus menjadi titik asal
    assert np.allclose(normalized[0, 63:66], 0.0, atol=1e-6)


def test_scale_normalization_has_unit_max_distance() -> None:
    data = _make_dummy_single_frame()
    step1 = normalize_landmarks_wrist_relative(data)
    step2 = normalize_landmarks_scale(step1)

    # Jarak tangan kanan dari pergelangan tangan
    right = step2[0, 0:63].reshape(-1, 3)
    d_right = np.linalg.norm(right - right[0], axis=1)
    if not np.allclose(d_right, 0.0):
        assert np.isclose(d_right.max(), 1.0, atol=1e-5)

    # Jarak tangan kiri dari pergelangan tangan
    left = step2[0, 63:126].reshape(-1, 3)
    d_left = np.linalg.norm(left - left[0], axis=1)
    if not np.allclose(d_left, 0.0):
        assert np.isclose(d_left.max(), 1.0, atol=1e-5)


def test_scale_normalization_avoids_division_by_small_scale() -> None:
    """Normalisasi skala harus no-op ketika semua landmark bertepatan dengan pergelangan tangan.

    Ini memvalidasi penjagaan terhadap max_distance yang sangat kecil di
    normalize_landmarks_scale sehingga kita tidak memperkuat noise numerik.
    """
    # Satu frame di mana semua landmark tangan kanan identik (tidak ada penyebaran)
    data = np.zeros((1, 126), dtype=np.float32)
    # Atur pergelangan tangan kanan
    data[0, 0:3] = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    # Salin pergelangan tangan ke beberapa titik lain di tangan yang sama
    for i in range(3, 21 * 3, 3):
        data[0, i:i+3] = data[0, 0:3]

    # Tangan kiri dibiarkan sebagai nol (tidak ada tangan terdeteksi)

    step1 = normalize_landmarks_wrist_relative(data)
    step2 = normalize_landmarks_scale(step1)

    # Setelah relatif pergelangan tangan, semua landmark tangan kanan harus tepat nol
    assert np.allclose(step1[0, 0:63], 0.0, atol=1e-7)
    # Normalisasi skala harus menjaganya tetap nol dan terbatas (tidak ada NaN / inf)
    assert np.allclose(step2[0, 0:63], 0.0, atol=1e-7)
    assert np.all(np.isfinite(step2[0, 0:63]))


def test_full_normalization_combines_steps() -> None:
    data = _make_dummy_single_frame()
    full = normalize_landmarks_full(data)

    # Masih berpusat pada pergelangan tangan
    assert np.allclose(full[0, 0:3], 0.0, atol=1e-6)
    assert np.allclose(full[0, 63:66], 0.0, atol=1e-6)

    # Harus mempertahankan beberapa varians (tidak semua nol)
    assert full.std() > 0.0


def test_zscore_normalization_3d_keeps_shape_and_scales() -> None:
    """Normalisasi Z-score untuk landmark 3D (126 fitur) menjaga bentuk dan skala per sumbu."""
    data = _make_dummy_single_frame().astype(np.float32)
    flat = data.reshape(-1)  # 126 fitur

    normalized = normalize_landmarks(flat, method="zscore")

    assert normalized.shape == flat.shape

    coords = normalized.reshape(-1, 3)
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)

    assert np.allclose(mean, 0.0, atol=1e-5)
    assert np.allclose(std, 1.0, atol=1e-5)


def test_zscore_normalization_2d_keeps_shape_and_scales() -> None:
    """Normalisasi Z-score untuk landmark 2D (84 fitur) menjaga bentuk dan skala per sumbu."""
    # Buat vektor landmark 2D dummy: 21 poin per tangan, 2 koordinat masing-masing, 2 tangan → 84 fitur
    num_points_per_hand = 21
    num_hands = 2
    num_dims = 2

    coords = np.stack([
        np.linspace(0.1, 0.9, num_points_per_hand * num_hands),
        np.linspace(0.2, 0.8, num_points_per_hand * num_hands),
    ], axis=1).astype(np.float32)

    flat = coords.reshape(-1)  # 84 fitur

    normalized = normalize_landmarks(flat, method="zscore")

    assert normalized.shape == flat.shape

    coords_norm = normalized.reshape(-1, num_dims)
    mean = coords_norm.mean(axis=0)
    std = coords_norm.std(axis=0)

    # Izinkan toleransi yang sedikit lebih longgar di sini untuk mengakomodasi pembulatan float32
    # sambil tetap menegakkan penskalaan zero-mean, unit-variance yang tepat.
    assert np.allclose(mean, 0.0, atol=1e-4)
    assert np.allclose(std, 1.0, atol=1e-4)


def test_zscore_normalization_3d_handles_constant_input() -> None:
    """Normalisasi Z-score untuk input 3D konstan harus terbatas dan nol.

    Ini melatih istilah penjaga 1e-7 yang ditambahkan ke std per sumbu untuk menghindari
    pembagian dengan nol ketika semua koordinat identik.
    """
    flat = np.full(126, 0.5, dtype=np.float32)

    normalized = normalize_landmarks(flat, method="zscore")

    # Bentuk dipertahankan dan semua nilai harus terbatas
    assert normalized.shape == flat.shape
    assert np.all(np.isfinite(normalized))

    # Untuk vektor konstan, z-score dengan epsilon harus menghasilkan semua nol
    assert np.allclose(normalized, 0.0, atol=1e-6)


def test_zscore_normalization_2d_handles_constant_input() -> None:
    """Normalisasi Z-score untuk input 2D konstan harus terbatas dan nol."""
    # Vektor datar 2D 84-fitur (misalnya 21 poin × 2 koordinat × 2 tangan)
    flat = np.full(84, -1.25, dtype=np.float32)

    normalized = normalize_landmarks(flat, method="zscore")

    assert normalized.shape == flat.shape
    assert np.all(np.isfinite(normalized))
    assert np.allclose(normalized, 0.0, atol=1e-6)


def run_all() -> None:
    test_wrist_relative_centers_on_wrist()
    test_scale_normalization_has_unit_max_distance()
    test_scale_normalization_avoids_division_by_small_scale()
    test_full_normalization_combines_steps()
    test_zscore_normalization_3d_keeps_shape_and_scales()
    test_zscore_normalization_2d_keeps_shape_and_scales()
    test_zscore_normalization_3d_handles_constant_input()
    test_zscore_normalization_2d_handles_constant_input()
    print("\n✅ test_unit_normalization_invariants.py: all tests passed")


if __name__ == "__main__":
    run_all()
