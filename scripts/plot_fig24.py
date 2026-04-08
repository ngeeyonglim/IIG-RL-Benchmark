import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "results/test/iem_e_ppo/classical_phantom_ttt"
latest = sorted(os.listdir(base_dir))[-1]
csv_path = os.path.join(base_dir, latest, "train_log.csv")
print(f"Reading: {csv_path}")

df = pd.read_csv(csv_path)

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.plot(df["steps"], df["eps_states_with_dead_fraction"])
ax.set_xlabel("Training Steps")
ax.set_ylabel("Fraction of States with Dead Actions")
ax.set_title("Fraction of Information States Containing Dead Actions")

plt.tight_layout()
plt.savefig("figures/fig24_dead_action_fraction.png")
print("Saved figures/fig24_dead_action_fraction.png")
