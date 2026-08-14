import numpy as np
from scipy.stats import skewnorm
for a in [-3, 0, 3]:
    mean, var, skew = skewnorm.stats(a, moments='mvs')
    print(f"Alpha={a}: mean={mean:.3f}, skewness={skew:.3f}")
