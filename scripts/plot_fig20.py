import matplotlib.pyplot as plt
import numpy as np

data = {
    "Tsallis\nIEM PPO": [0.12275, 0.1157, 0.14925],
    "Tsallis\nPPO": [0.13799163, 0.152302784, 0.168787506],
    "Shannon\nIEM PPO": [0.121614618, 0.128651624, 0.145490435],
    "Shannon\nPPO": [0.123081213, 0.106080519, 0.195110635],
}

labels = list(data.keys())
values = list(data.values())
means = [np.mean(v) for v in values]

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.boxplot(values, tick_labels=labels)

for i, mean in enumerate(means, start=1):
    ax.text(i, mean, f"{mean:.4f}", ha="center", va="center", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

ax.set_ylabel("Final avg exploitability")
ax.set_title("Final exploitability across algorithms")

plt.tight_layout()
plt.savefig("figures/fig20_algorithm_comparison.png")
print("Saved figures/fig20_algorithm_comparison.png")
