import numpy as np

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    print(np.rad2deg((ang1 - ang2) % (2 * np.pi)))


horizon = (0, 1)

m = np.array([694, 663])
p = np.array([693, 651])
v = p-m
angle_between(v, horizon)