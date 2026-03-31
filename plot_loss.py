import matplotlib.pyplot as plt

# 🔥 YOUR TRAINING VALUES (already known)
base_loss = [145.6067, 145.4997, 145.2904]
dis_loss = [145.9931, 145.4378, 145.3742]

epochs = [1, 2, 3]

plt.figure()

plt.plot(epochs, base_loss, marker='o', label="Baseline")
plt.plot(epochs, dis_loss, marker='o', label="Disentangled")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")

plt.legend()
plt.grid()

plt.savefig("loss_comparison.png")
plt.show()