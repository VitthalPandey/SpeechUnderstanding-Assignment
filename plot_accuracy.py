import matplotlib.pyplot as plt

# 🔥 FINAL RESULTS (your values)
models = ["Baseline", "Disentangled"]
accuracy = [0.10, 0.10]

plt.figure()

plt.bar(models, accuracy)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Comparison: Baseline vs Disentangled")

# annotate values on bars
for i, v in enumerate(accuracy):
    plt.text(i, v + 0.002, f"{v:.2f}", ha='center')

plt.ylim(0, 0.2)
plt.grid()

plt.savefig("accuracy_comparison.png")
plt.show()