from numpy import load
import matplotlib.pyplot as plt

before = load("before.npy")
after = load("after.npy")
resampled = load("resampled.npy")

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
print(before)
print(after)
print(resampled)
ax1.hist(before[0])
ax2.hist(after[0])
ax3.hist(resampled[0])
plt.show()
