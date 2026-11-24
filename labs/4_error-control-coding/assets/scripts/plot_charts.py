from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_traffic_comparison(
    p_dense: np.ndarray,
    arq_theory_mb: np.ndarray,
    probs: np.ndarray,
    arq_bytes_mb: np.ndarray,
    ham_bytes_mb: np.ndarray,
    orig_mb: float,
    output_path: Path,
) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        p_dense,
        arq_theory_mb,
        "--",
        color="tab:blue",
        alpha=0.7,
        label="Теория: бит четности + повторный запрос",
    )
    ax.plot(
        probs,
        arq_bytes_mb,
        "o-",
        color="tab:blue",
        label="Эксперимент: бит четности + повтор (Задание 1)",
    )
    ax.plot(
        probs,
        ham_bytes_mb,
        "s-",
        color="tab:orange",
        label="Эксперимент: код Хэмминга (7, 4) (Задание 2)",
    )
    ax.axhline(
        orig_mb,
        linestyle=":",
        color="gray",
        label="Исходный объем файла (без кодирования)",
    )

    p_star = 0.065
    ax.axvline(p_star, linestyle="-.", color="tab:red", alpha=0.7)
    ax.annotate(
        "Точка равенства трафика\n$p^* \\approx 0,065$",
        xy=(p_star, orig_mb * 1.75),
        xytext=(0.11, orig_mb * 1.45),
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
        color="tab:red",
    )

    # ax.set_title("Зависимость объема переданных данных от вероятности ошибки $p$")
    ax.set_xlabel("Вероятность битовой ошибки в канале, $p$")
    ax.set_ylabel("Объем переданных данных, млн байт")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_path.suffix.lstrip(".").lower() or "svg"
    fig.savefig(output_path, format=fmt, bbox_inches="tight")
    plt.close(fig)


def plot_residual_errors(
    probs: np.ndarray,
    arq_ber: np.ndarray,
    ham_ber: np.ndarray,
    output_path: Path,
) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        probs,
        arq_ber,
        "o-",
        color="tab:blue",
        label="Бит четности + повтор (необнаруженные четные ошибки)",
    )
    ax.plot(
        probs,
        ham_ber,
        "s-",
        color="tab:orange",
        label="Код Хэмминга (7, 4) (кратные ошибки $\\geq 2$ в блоке)",
    )

    # ax.set_title("Зависимость остаточной доли искаженных байт от вероятности $p$")
    ax.set_xlabel("Вероятность битовой ошибки в канале, $p$")
    ax.set_ylabel("Доля неверно принятых байт, %")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_path.suffix.lstrip(".").lower() or "svg"
    fig.savefig(output_path, format=fmt, bbox_inches="tight")
    plt.close(fig)
