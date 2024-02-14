import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

amplitudes = []
for snapshot_number in tqdm(range(300)):
    snapshots = np.load("dataset/gprmax_output_files/scan_" + str(snapshot_number).zfill(5) + "/snapshots.npz")["00000_E"]

    l = []
    for snap in snapshots:
        max = np.abs(snap).max()
        l.append(max)

    amplitudes.append(l)

amplitudes = np.asarray(amplitudes)
amplitudes = amplitudes.mean(axis=0)

def f(x, a, b, c):
    return a/(x+b) + c

def g(x, a, b, c):
    return a * np.e**(b*(x + c))

x_data = np.linspace(2, 25, 24)
y_data = amplitudes

# popt_f, _ = curve_fit(f, x_data, y_data, bounds=([0, 0, -85], np.inf))
popt_f, _ = curve_fit(f, x_data, y_data)
popt_g, _ = curve_fit(g, x_data, y_data)

print("F params:", popt_f)
print("G params:", popt_g)


print("F last:", f(25, *popt_f))
print("G last:", g(25, *popt_g))

def get_time_weight(t):
    return 252.86 * np.e**(-0.2388*(t + -4.8314))

plt.plot(x_data, amplitudes)
# plt.plot(x_data, g(x_data, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.plot(x_data, f(x_data, *popt_f), 'r-', label='f: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_f))
plt.plot(x_data, g(x_data, *popt_g), 'g-', label='g: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_g))
# plt.plot(x_data, get_time_weight(x_data), 'cyan', label='g2')
plt.legend()
plt.show()


level_f = y_data / f(x_data, *popt_f)
level_g = y_data / g(x_data, *popt_g)

plt.plot(level_f, "r-", label="f")
plt.plot(level_g, "g-", label="g")
plt.legend()
ax = plt.gca()
ax.set_ylim([0, 2])
plt.show()