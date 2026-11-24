from pathlib import Path
from channel_coding_core import load_war_peace, run_parity_arq


def main():
    script_dir = Path(__file__).resolve().parent
    assets_dir = script_dir.parent
    data_dir = assets_dir / "data"

    data = load_war_peace(data_dir / "war_peace.txt")
    sample_size = min(len(data), 50_000)
    data_sample = data[:sample_size]

    test_probs = [0.0, 0.01, 0.05, 0.1]
    print(f"Размер исследуемого фрагмента: {sample_size} байт\n")

    for p in test_probs:
        stats = run_parity_arq(data_sample, p_error=p, seed=42)
        k_expansion = stats.equivalent_bytes / stats.source_bytes
        residual_pct = (
            stats.residual_byte_errors / stats.source_bytes
        ) * 100.0

        print(
            f"p = {p:.2f} | Передано байт: {stats.equivalent_bytes:8.1f} | "
            f"Коэфф. расширения: {k_expansion:.3f} | "
            f"Повторов: {stats.retransmissions:6d} | "
            f"Остаточных ошибок: {stats.residual_byte_errors:5d} ({residual_pct:.2f}%)"
        )


if __name__ == "__main__":
    main()
