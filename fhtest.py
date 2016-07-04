# Test Case for the Falk-Hoffman Algorithm. In the following, the two examples from
# James E. Falk, Karla L. Hoffman: "Concave Minimization via Collapsing Polytopes", Operations Research 34.6 (1986)
# are implemtented and the solutions are returned

from FalkHoffmanAlgorithm import *
import numpy as np

# Example 1
print 'Solving Example 1:'
t = FalkHoffmanInstance(
    f=lambda x: -(x[0]-2)**2-(x[1]-2)**2,
    A=np.matrix([
        [-1, -1],
        [1, -2],
        [2, -1],
        [3, 5],
        [-6, 10],
        [-1, 0],
        [0, -1]
    ]),
    b=np.array([-1, 1, 5, 27, 30, 0, 0])
)

# start solver
opt, xCoords, status = t.solve()
print 'Optimal Target Function Value: ' + str(opt)
print 'Coordinates corresponding to optimal target function values:'
print xCoords

# Example 2
print 'Solving Example 2:'
t = FalkHoffmanInstance(
    f=lambda x: -(x[0]-1)**2-x[1]**2-(x[2]-1)**2,
    A=np.matrix([
        [-4, 5, -4],
        [-6, 1, 1],
        [1, 1, -1],
        [12, 5, 12],
        [12, 12, 7],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ]),
    b=np.array([-4, -4.1, 1, 34.8, 29.1, 0, 0, 0])
)

# start solver
opt, xCoords, status = t.solve()

print 'Optimal Target Function Value: ' + str(opt)
print 'Coordinates corresponding to optimal target function values:'
print xCoords

