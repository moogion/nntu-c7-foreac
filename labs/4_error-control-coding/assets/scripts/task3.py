import csv
from pathlib import Path
import numpy as np

from channel_coding_core import (
    load_war_peace,
    simulate_hamming_74_fast,
    simulate_parity_arq_fast,
)
from plot_charts import plot_residual_errors, plot_traffic_comparison


def main():
    script_dir = Path(__file__).resolve().parent
    assets_dir = script_dir.parent
    data_dir = assets_dir / "data"
    images_dir = assets_dir / "images"

    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    data = load_war_peace(data_dir / "war_peace.txt")
    raw_bytes = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(raw_bytes).reshape(-1, 8)
    n_bytes = len(raw_bytes)
    orig_mb = n_bytes / 1e6

    rng = np.random.default_rng(2026)

    key_probs = [0.0, 0.01, 0.05, 0.065, 0.1, 0.2, 0.3]
    probs = np.unique(np.concatenate([np.linspace(0.0, 0.45, 19), key_probs]))
    probs.sort()

    arq_bytes_dict = {}
    arq_ber_dict = {}
    ham_bytes_dict = {}
    ham_ber_dict = {}

    for p in probs:
        b_arq, e_arq = simulate_parity_arq_fast(bits, float(p), rng)
        b_ham, e_ham = simulate_hamming_74_fast(bits, float(p), rng)
        arq_bytes_dict[p] = b_arq
        arq_ber_dict[p] = e_arq * 100.0
        ham_bytes_dict[p] = b_ham
        ham_ber_dict[p] = e_ham * 100.0

    p_dense = np.linspace(0.0, 0.45, 300)
    arq_theory_mb = (n_bytes * 2.25 / (1.0 + (1.0 - 2.0 * p_dense) ** 9)) / 1e6

    chart1_path = images_dir / "bytes_vs_error_prob.svg"
    chart2_path = images_dir / "residual_error_vs_prob.svg"

    plot_traffic_comparison(
        p_dense=p_dense,
        arq_theory_mb=arq_theory_mb,
        probs=probs,
        arq_bytes_mb=np.array([arq_bytes_dict[p] for p in probs]) / 1e6,
        ham_bytes_mb=np.array([ham_bytes_dict[p] for p in probs]) / 1e6,
        orig_mb=orig_mb,
        output_path=chart1_path,
    )

    plot_residual_errors(
        probs=probs,
        arq_ber=np.array([arq_ber_dict[p] for p in probs]),
        ham_ber=np.array([ham_ber_dict[p] for p in probs]),
        output_path=chart2_path,
    )

    print(f"График объема переданных данных сохранен: {chart1_path.relative_to(assets_dir.parent)}")
    print(f"График остаточных ошибок сохранен: {chart2_path.relative_to(assets_dir.parent)}")

    csv_rows = []

    print("\nСводная таблица результатов моделирования (war_peace.txt):")
    print(f"{'p':<8} | {'ARQ (байт)':<14} | {'Коэфф. ARQ':<10} | {'Хэмминг (байт)':<14} | {'Ошибки ARQ / Хэм., %':<20}")
    print("-" * 75)

    for kp in key_probs:
        b_a = arq_bytes_dict[kp]
        k_a = b_a / n_bytes
        b_h = ham_bytes_dict[kp]
        e_a = arq_ber_dict[kp]
        e_h = ham_ber_dict[kp]

        csv_rows.append([
            f"{kp:.3f}",
            f"{int(round(b_a))}",
            f"{k_a:.3f}",
            f"{int(round(b_h))}",
            f"{e_a:.2f}",
            f"{e_h:.2f}",
        ])
        print(
            f"{kp:<8.3f} | {int(round(b_a)):<14} | {k_a:<10.3f} | {int(round(b_h)):<14} | {e_a:.2f} / {e_h:.2f}"
        )

    csv_path = data_dir / "simulation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "error_prob_p",
            "arq_bytes",
            "arq_expansion_ratio",
            "hamming_bytes",
            "arq_residual_error_pct",
            "hamming_residual_error_pct",
        ])
        writer.writerows(csv_rows)

    print(f"\nДанные таблицы экспортированы в: {csv_path.relative_to(assets_dir.parent)}")


if __name__ == "__main__":
    main()
