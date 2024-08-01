import numpy as np

file = np.load("./global_route_town04.npy")
print(file)
np.savetxt('./data.txt', file)