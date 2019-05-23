import numpy as np
import time

a = np.array([1,2,3,4])

b = np.random.rand(1000000)
c = np.random.rand(1000000)
tic = time.time()
d = np.dot(b,c)
toc = time.time()
print(d)
print("Vectorization version: " + str(1000*(toc-tic)) + "ms")

d = 0
tic = time.time()
for i in range(1000000):
    d += b[i]*c[i]
toc = time.time()
print(d)
print("For loop: " + str(1000*(toc-tic)) + "ms")