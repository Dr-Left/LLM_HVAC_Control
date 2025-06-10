import matplotlib.pyplot as plt
import numpy as np

# Apply professional plotting style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 28,
        "axes.labelsize": 22,
        "axes.titlesize": 28,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
        "lines.linewidth": 2.5,
        "grid.alpha": 0.3,
        "figure.titlesize": 32,
        "font.family": ["Arial", "Songti SC"],
    }
)

t_interval = [180, 360, 3600]
reward = [1.22, 1.24, 0.81]

plt.figure(figsize=(8, 6))
plt.plot(reward, marker="o", linestyle="-", color="darkblue", linewidth=2.5)
plt.xlabel("时间间隔 (s)", fontweight="bold")
plt.ylabel("奖励值", fontweight="bold")
plt.title("奖励值与仿真时间步长(s)的关系", fontweight="bold", pad=15)

plt.xticks([0, 1, 2], ["180", "360", "3600"])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reward_vs_time_interval.pdf")
