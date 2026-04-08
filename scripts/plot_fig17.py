import pandas as pd
import matplotlib.pyplot as plt

labels = ["MEC PPO", "Standard PPO"]
values = [
    (1.568355109744809 + 0.0021531902136390224) / 2,
    (0.0075131560388109 + 0.0001423439850319) / 2,
]

fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
bars = ax.bar(labels, values)
ax.set_yscale("log")
ax.set_ylabel("Final Avg Exploitability")
ax.set_title("MEC PPO vs Standard PPO")

for bar, val in zip(bars, values):
    ax.annotate(
        f"{val:.4f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.savefig("figures/fig17_mec_vs_ppo.png")
print("Saved figures/fig17_mec_vs_ppo.png")
