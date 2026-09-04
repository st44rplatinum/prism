import main
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
bins = 40

plt.hist(main.single_sample["score"], bins=bins, alpha=0.5, label="Single Mutations", color="blue")
plt.hist(main.double_sample["score"], bins=bins, alpha=0.5, label="Double Mutations", color="orange")
plt.title("Distribution of Scores for Single and Double Mutations")

plt.xlabel("Target value (score)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()