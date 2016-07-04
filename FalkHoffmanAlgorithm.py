import numpy as np
from pulp import *
import collections


class FalkHoffmanInstance(object):
    """
    Solve concave optimization problem on convex solution domain, applying the Falk-Hoffman Algorithm
    [FalkHoffman1986]_. Notation inspired by that source.

    References
    ----------
    .. [FalkHoffman1986] James E. Falk, Karla L. Hoffman: "Concave Minimization via Collapsing Polytopes",
        Operations Research 34.6 (1986). DOI: 10.1287/opre.34.6.919
    """
    class solutionTreeNode(object):
        """
        Object representing one node in the solution tree
        """
        def __init__(self, problemInstance, x, y, w, s):
            """
            Constructor

            Parameters
            ----------
            problemInstance : FalkHoffmanInstance
                owning problem instance
            x : numpy.array
                node coordinates in the original optimization problem
            y : float
                distance value (f) for point v
            w : numpy.array
                feasible point associated with v
            s : numpy.array
                slack variable vector, corresponding to v
            """
            self._problemInstance = problemInstance
            # compose simplex tableau solution vector
            self._solution = self._problemInstance.combineSolutionVector(x, y, s)

            # we now perform a sanity check on the solution vector. In particular, the number of
            # 0 values in the solution vector has to be n+1.
            k = self._problemInstance.A.shape[1]+1
            if np.sum(self._solution == 0) < k:
                # partition set of indices into, split by n+1 smallest element
                k_smallest_indices = np.argpartition(self._solution, k)
                # set the values corresponding to the n+1 smallest values to 0
                np.put(self._solution, k_smallest_indices[:k], 0)
                # re-convert the solution into three components
                self._x, self._y, self._s = self._problemInstance.getSolutionComponents(self._solution)
            else:  # no adjustment necessary, we will just use the values passed
                # store coordinates and values associated with this point
                self._x = x
                self._w = w
                self._y = y
                self._s = s

            # check if solution is feasible
            self._feasible = True if np.isclose(self._y, 0) else False
            # define some additional variables
            self._children = []
            self._f = self._problemInstance.f(w)  # calculate target function value at w

            # register node with problem instance
            self._problemInstance.registerSolutionNode(self)

            print 'New Node Created representing ' + str(self._x) + ' with f-Value: ' + str(self._f)

        def getLowestTargetFunctionLeaf(self):
            """
            Search tree for leaf with lowest value for target function value
            """
            if self.isLeaf:
                return self
            else:  # return minimum of direct children responses
                currentLeaf = None
                currentMinValue = np.inf
                for c in self._children:
                    childBestLeaf = c.getLowestTargetFunctionLeaf()
                    if childBestLeaf.f < currentMinValue:
                        currentLeaf = childBestLeaf
                        currentMinValue = childBestLeaf.f
                return currentLeaf

        def branchOut(self):
            """
            Create all direct child-nodes of the current solution node, performing all checks
            on the elgibility of a possible new solution node
            """
            # get binary indicator about which solution variables are in the basis
            isBaseVector = (self._solution > 0)
            # get matrix of Basis and non-basis columns from tableau
            t = self._problemInstance.originalTableau
            # get Matrix comprising of Basis Vectors and it's inverse, required for pivoting operation
            B = t[:, isBaseVector]
            B_inv = np.linalg.inv(B)

            # calculate non-basis vectors in current simplex tableau
            N = B_inv.dot(t[:, ~isBaseVector])

            # compute and store current rhs tableau vector to speed up computation
            currentRhs = B_inv.dot(self._problemInstance.b)

            # Find row index corresponding to y
            # first calculate current tableau column corresponding to y
            yCol = B_inv.dot(t[:, self._problemInstance.A.shape[1]])
            # find only index of yCol close to 1
            yRow = np.argmax(np.isclose(yCol, 1))

            # iterate through all column indices of non-basis vectors
            # N_col will allow us to directly retrieve the corresponding column vector from N, saving
            # computation time
            for N_col, j in enumerate(np.array(range(t.shape[1]))[~isBaseVector]):
                # get corresponding tableau column
                col = N[:, N_col]
                # implement first check for new nodes: A_ij > 0.
                # entry in the entering variable column, in the row corresponding to y has to be > 0
                if col[yRow] <= 0:
                    continue
                # calculate possible theta values
                thetaVector = np.divide(
                    currentRhs,
                    col
                )

                # create a solution vector for the new solutions
                # first for new point v
                solution_new_v = self._solution.copy()
                solution_new_v[isBaseVector] = currentRhs-np.min(thetaVector[thetaVector > 0])*col
                solution_new_v[j] = np.min(thetaVector[thetaVector > 0])
                v_x, v_y, v_s = self._problemInstance.getSolutionComponents(solution_new_v)
                # if y has improved, we also need to check, if the new x-coordinates are not
                # already part of the solution tree
                if self._problemInstance.solutionNodeExists(v_x):
                    continue
                # if both tests have been passed, calculate the corresponding w-point
                solution_new_w = self._solution.copy()
                solution_new_w[isBaseVector] = currentRhs-thetaVector[yRow]*col
                solution_new_w[j] = thetaVector[yRow]
                w_x, w_y, w_s = self._problemInstance.getSolutionComponents(solution_new_w)

                # create child node for this new solution
                self._children.append(
                    FalkHoffmanInstance.solutionTreeNode(
                        self._problemInstance,
                        v_x,
                        v_y,
                        w_x,
                        v_s
                    )
                )

        def setF(self, newValue):
            """
            Allow to externally set target function value.
            External changes are only accepted for the root node, where this change is required by the
            algorithm.
            """
            # only allow changes at root node
            if self == self._problemInstance.solutionTreeRoot:
                self._f = newValue

        isLeaf = property(
            lambda self: len(self._children) == 0,
            lambda self, newValue: False
        )
        """Read only property: Direct child nodes"""
        x = property(
            lambda self: self._x,
            lambda self, newValue: False
        )
        """Read-only property: x-Coordinates"""
        y = property(
            lambda self: self._y,
            lambda self, newValue: False
        )
        """Read-only property: y-Value (distance to nearest constraint)"""
        f = property(
            lambda self: self._f,
            lambda self, newValue: self.setF(newValue)
        )
        """Read/Write access to target function value. External changes to target function value are
        only permissible for the root node"""
        feasible = property(
            lambda self: self._feasible,
            lambda self, newValue: False
        )
        """Read only property: Is this solution feasible (y==0)"""

    def __init__(self, f, A, b):
        """
        Parameters
        ----------
        f : callable
            Function assumed to be concave. Called with numpy vector, representing a possible solution vector x.
            Expcted to return float.
        A : numpy.Matrix
            m x n Matrix, representing the m inequality constraints (no slack variables). Should include non-negativity
            constraints
        b : numpy.array
            Right-hand-side vector (such that Ax<=b)
        """
        # store problem parameters
        self._f = f
        self._A = np.array(A)
        self._b = b
        # row wise euclidean norm, c.f. http://stackoverflow.com/a/7741976
        self._a = np.sum(np.abs(self._A)**2, axis=-1)**(1./2)
        # create original Simplex tableau
        self._originalTableau = self.getInitialTableau()
        # create dictionary to quickly check the existence of a solution node for a particular coordinate
        self._solutionNodeDict = {}

        # solve initial CP and create root node of solution tree
        x, y, s = self.solveCP()
        self._solutionTreeRoot = FalkHoffmanInstance.solutionTreeNode(
            self,
            x,
            y,
            x,  # v and w are the same point here
            s
        )
        self._solutionTreeRoot.f = np.inf  # set target function value artificially to +infty

    def registerSolutionNode(self, node):
        """
        Make problem aware of the existence of a new node in the solution tree

        Parameters
        ----------
        node : solutionTreeNode
        """
        self._solutionNodeDict[tuple(node.x)] = node

    def solutionNodeExists(self, x):
        """
        Check if a solution node to coordinate tuple x exist already.

        Parameters
        ----------
        x : numpy.array
            coordinates

        Returns
        -------
        c : Boolean
            Result
        """
        return tuple(x) in self._solutionNodeDict.keys()

    def combineSolutionVector(self, x, y, s):
        """
        Create combined solution vector, encompassing the solution point (x), the distance value y
        and the slack variable vector s.

        Parameters
        ----------
        x : numpy.array
        y : float
        s : numpy.array

        Returns
        -------
        solution : numpy.array
        """
        solution = np.append(x, y)
        return np.append(solution, s)

    def getSolutionComponents(self, solution):
        """
        Dissembles a solution point to the simplex tableau v into the components x (point in the original
        decision problem), y (distance to constraints), and s (slack variable vector).

        Parameters
        ----------
        solution : numpy.float

        Returns
        -------
        x : numpy.array
        y : float
        s : numpy.array
        """
        x = solution[:self._A.shape[1]]
        y = solution[self._A.shape[1]]
        s = solution[self._A.shape[1]+1:]
        return x, y, s

    def getInitialTableau(self):
        """
        Compose original (before any optimization) simplex tableau (no right-hand side vector)

        Returns
        -------
        t : numpy.matrix
        """
        # tableau has has many rows as A and one column for each column of A, y and one slack variable
        # for each constraint
        t = np.zeros((self._A.shape[0], self._A.shape[1]+1+self._A.shape[0]))
        # the left most part of m is A
        t[:, :self._A.shape[1]] = self._A
        # next row is the length vector a
        t[:, self._A.shape[1]] = self._a
        # the other columns are an identity matrix
        t[:, self._A.shape[1]+1:] = np.identity(self._A.shape[0])
        return t

    def solve(self, maxK=10000):
        """
        Start solver.

        Parameters
        ----------
        maxK : int
            Maximum number of stages to be computed. Default: 10,000

        Returns
        -------
        y : float
            best feasible solution found
        l : list
            list of numpy arrays, vectors that yield the best found target function value
        status : str
            Status string indicating the type of solution found. 'optimal' is returned if the algorithm
            concluded as planned and optimality of the solution can be guaranteed. 'stopped' is returned
            if the algorithm hit the maximum number of stages and was terminated.
        """
        k = 0
        status = 'stopped'
        while k <= maxK:
            k += 1  # increase stage counter
            n = self._solutionTreeRoot.getLowestTargetFunctionLeaf()
            if np.isclose(n.y, 0):  # this is a feasible solution, we have finished
                status = 'optimal'
                break
            else:
                n.branchOut()

        # the tree has stopped for whatever reason. Now we need to find leafs corresponding to
        # the optimal target function values
        optimalNodes = []
        optimalTargetFunctionValue = np.inf
        for x, node in self._solutionNodeDict.iteritems():
            if node.feasible:
                if np.isclose(node.f, optimalTargetFunctionValue):
                    optimalNodes.append(x)
                elif node.f < optimalTargetFunctionValue:
                    optimalNodes = [x]
                    optimalTargetFunctionValue = node.f

        return optimalTargetFunctionValue, optimalNodes, status

    def solveCP(self):
        """
        Solve Linear Optimization problem CP.

        Returns
        -------
        x : numpy.array
            Optimal point
        y : float
            maximum achieved sphere radius
        s : numpy.array
            Slack variable values at optimality
        """
        prob = LpProblem("FHAlgorithmCP", LpMaximize)
        # create solution variables
        x = LpVariable.dicts(
            "x",
            indexs=range(self._A.shape[1]),
            lowBound=0
        )
        y = LpVariable(
            "y",
            lowBound=0
        )
        s = LpVariable.dicts(
            "s",
            indexs=range(self._A.shape[0]),
            lowBound=0
        )
        # target function: minimize weighted set size of active worksystems
        prob += y, "Target Function"
        # main constraints (Ax+ay+s = b)
        for i in range(self._A.shape[0]):
            prob += lpSum(
                [
                    self._A[i, j] * x[j]
                    for j in range(self._A.shape[1])
                ]
            ) + self._a[i]*y + s[i] == self._b[i], 'Constraint ' + str(i)
        # make sure y is minimum value (y <= (b_i-A(i)x)/||A(i)|| forall i)
        for i in range(self._A.shape[0]):
            prob += y <= (
                self._b[i] - lpSum(
                    [
                        self._A[i, j] * x[j]
                        for j in range(self._A.shape[1])
                    ]
                )
            )/self._a[i], 'y-minimization constraint, iteration ' + str(i)

        # save and solve LP
        prob.writeLP("FHAlgorithmCP.lp")
        prob.solve()

        if LpStatus[prob.status] == 'Optimal':  # solution found
            # return x, y, s as arrays/floats respectively
            xRet = np.array([v.value() for v in x.values()])
            yRet = y.value()
            sRet = np.array([v.value() for v in s.values()])

            # delete all temporary files created by the solver (lp and mps files)
            for f in [f for f in os.listdir(".") if f.endswith(".mps") or f.endswith(".lp") or f.endswith(".sol")]:
                os.remove(f)

            return xRet, yRet, sRet
        else:
            assert False, 'Initial CP could not be solved at optimality'

    originalTableau = property(
        lambda self: self._originalTableau,
        lambda self, newValue: False
    )
    """read-only access on original simplex tableau"""
    f = property(
        lambda self: self._f,
        lambda self, newValue: False
    )
    """read-only access on target function"""
    A = property(
        lambda self: self._A,
        lambda self, newValue: False
    )
    """read-only access on constraint matrix"""
    b = property(
        lambda self: self._b,
        lambda self, newValue: False
    )
    """read-only access on right-hand side vector"""
    solutionTreeRoot = property(
        lambda self: self._solutionTreeRoot,
        lambda self, newValue: False
    )
    """read-only access on root node of solution tree"""
