import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
for a in [-3, 0, 3]:
    pdf = skewnorm.pdf(x, a, loc=0, scale=1)
    mode_x = x[np.argmax(pdf)]
    mean = skewnorm.mean(a)
    print(f"Alpha={a:2d}: Mode={mode_x:.2f}, Mean={mean:.2f}")
