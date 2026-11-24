from dataclasses import dataclass
from pathlib import Path
import numpy as np
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/NNTU-CS/RECF-2026/main/Data/war_peace.txt"
)


@dataclass(frozen=True)
class TransmissionStats:
    source_bytes: int
    total_bits: int
    zeros_sent: int
    ones_sent: int
    retransmissions: int
    corrected_errors: int
    residual_byte_errors: int

    @property
    def equivalent_bytes(self) -> float:
        return self.total_bits / 8.0


def load_war_peace(path: Path | None = None) -> bytes:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "war_peace.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path.read_bytes()


def bytes_to_bitstrings(data: bytes) -> list[str]:
    return [f"{byte:08b}" for byte in data]


def add_parity_bit(byte_bits: str) -> str:
    parity = str(byte_bits.count("1") % 2)
    return byte_bits + parity


def check_parity(frame: str) -> bool:
    data_bits, parity_bit = frame[:8], frame[8]
    return (data_bits.count("1") % 2) == int(parity_bit)


def transmit_bsc_str(
    bit_str: str, p_error: float, rng: np.random.Generator
) -> str:
    if p_error <= 0.0:
        return bit_str
    flips = rng.random(len(bit_str)) < p_error
    return "".join(
        ("1" if ch == "0" else "0") if flip else ch
        for ch, flip in zip(bit_str, flips)
    )


def run_parity_arq(
    data: bytes, p_error: float, seed: int = 42
) -> TransmissionStats:
    rng = np.random.default_rng(seed)
    byte_strings = bytes_to_bitstrings(data)

    zeros_sent = 0
    ones_sent = 0
    retransmissions = 0
    residual_errors = 0

    for byte_bits in byte_strings:
        tx_frame = add_parity_bit(byte_bits)
        while True:
            zeros_sent += tx_frame.count("0")
            ones_sent += tx_frame.count("1")
            rx_frame = transmit_bsc_str(tx_frame, p_error, rng)

            if check_parity(rx_frame):
                if rx_frame[:8] != byte_bits:
                    residual_errors += 1
                break
            retransmissions += 1

    return TransmissionStats(
        source_bytes=len(data),
        total_bits=zeros_sent + ones_sent,
        zeros_sent=zeros_sent,
        ones_sent=ones_sent,
        retransmissions=retransmissions,
        corrected_errors=0,
        residual_byte_errors=residual_errors,
    )


def encode_hamming_74(nibble: str) -> str:
    d1, d2, d3, d4 = (int(ch) for ch in nibble)
    c1 = d1 ^ d2 ^ d4
    c2 = d1 ^ d3 ^ d4
    c4 = d2 ^ d3 ^ d4
    return f"{c1}{c2}{d1}{c4}{d2}{d3}{d4}"


def decode_hamming_74(rx_word: str) -> tuple[str, bool]:
    bits = [int(ch) for ch in rx_word]
    c1, c2, d1, c4, d2, d3, d4 = bits

    s1 = c1 ^ d1 ^ d2 ^ d4
    s2 = c2 ^ d1 ^ d3 ^ d4
    s4 = c4 ^ d2 ^ d3 ^ d4
    error_pos = s1 + 2 * s2 + 4 * s4

    corrected = False
    if error_pos != 0:
        bits[error_pos - 1] ^= 1
        corrected = True

    restored_nibble = f"{bits[2]}{bits[4]}{bits[5]}{bits[6]}"
    return restored_nibble, corrected


def run_hamming_74(
    data: bytes, p_error: float, seed: int = 42
) -> TransmissionStats:
    rng = np.random.default_rng(seed)
    byte_strings = bytes_to_bitstrings(data)

    zeros_sent = 0
    ones_sent = 0
    corrected_errors = 0
    residual_errors = 0

    for byte_bits in byte_strings:
        restored_parts: list[str] = []
        for nibble in (byte_bits[:4], byte_bits[4:]):
            tx_word = encode_hamming_74(nibble)
            zeros_sent += tx_word.count("0")
            ones_sent += tx_word.count("1")

            rx_word = transmit_bsc_str(tx_word, p_error, rng)
            dec_nibble, was_corrected = decode_hamming_74(rx_word)
            if was_corrected:
                corrected_errors += 1
            restored_parts.append(dec_nibble)

        if "".join(restored_parts) != byte_bits:
            residual_errors += 1

    return TransmissionStats(
        source_bytes=len(data),
        total_bits=zeros_sent + ones_sent,
        zeros_sent=zeros_sent,
        ones_sent=ones_sent,
        retransmissions=0,
        corrected_errors=corrected_errors,
        residual_byte_errors=residual_errors,
    )


def simulate_parity_arq_fast(
    bits: np.ndarray, p_error: float, rng: np.random.Generator
) -> tuple[float, float]:
    n_bytes = bits.shape[0]
    pending = np.arange(n_bytes)
    total_bits = 0
    undetected_errors = 0

    while pending.size > 0:
        total_bits += pending.size * 9
        if p_error <= 0.0:
            break
        flips = rng.random((pending.size, 9)) < p_error
        err_counts = np.sum(flips, axis=1)
        accepted_mask = (err_counts & 1) == 0
        undetected_errors += int(
            np.count_nonzero(err_counts[accepted_mask] > 0)
        )
        pending = pending[~accepted_mask]

    return total_bits / 8.0, undetected_errors / n_bytes


def simulate_hamming_74_fast(
    bits: np.ndarray, p_error: float, rng: np.random.Generator
) -> tuple[float, float]:
    n_bytes = bits.shape[0]
    nibbles = bits.reshape(-1, 4)
    d1, d2, d3, d4 = nibbles[:, 0], nibbles[:, 1], nibbles[:, 2], nibbles[:, 3]
    c1 = d1 ^ d2 ^ d4
    c2 = d1 ^ d3 ^ d4
    c4 = d2 ^ d3 ^ d4
    codewords = np.column_stack([c1, c2, d1, c4, d2, d3, d4])

    total_bits = codewords.size
    if p_error <= 0.0:
        return total_bits / 8.0, 0.0

    flips = rng.random(codewords.shape) < p_error
    rx = codewords ^ flips.astype(np.uint8)

    s1 = rx[:, 0] ^ rx[:, 2] ^ rx[:, 4] ^ rx[:, 6]
    s2 = rx[:, 1] ^ rx[:, 2] ^ rx[:, 5] ^ rx[:, 6]
    s4 = rx[:, 3] ^ rx[:, 4] ^ rx[:, 5] ^ rx[:, 6]
    err_pos = s1 + 2 * s2 + 4 * s4

    corr_mask = err_pos > 0
    rows = np.nonzero(corr_mask)[0]
    rx[rows, err_pos[corr_mask] - 1] ^= 1

    decoded_bytes = rx[:, [2, 4, 5, 6]].reshape(n_bytes, 8)
    residual_errors = int(
        np.count_nonzero(np.any(decoded_bytes != bits, axis=1))
    )
    return total_bits / 8.0, residual_errors / n_bytes
