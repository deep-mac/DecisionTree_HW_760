import numpy as np
import matplotlib.pyplot as plt

x1 = [0, 1]
y1 = [0, 1]

x2 = [0, 1]
y2 = [1, 0]

fig,ax = plt.subplots()
plt.scatter(x1, y1, c = "blue")
plt.scatter(x2, y2, c = "red")
plt.savefig('q2.pdf', format="pdf")


