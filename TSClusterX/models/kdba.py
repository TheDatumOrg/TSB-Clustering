import time
import numpy as np
from numpy.random import randint
from models.model import BaseClusterModel


def kDBA(X, k, max_iter=100, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    m = X.shape[0]
    idx = randint(0, k, size=m)
    cent = np.random.rand(k, X.shape[1])
    D = np.zeros((m, k))

    for it in range(max_iter):
        old_idx = idx

        # Update centroids using DBA (simplified without multiprocessing)
        for j in range(k):
            centroid_data = [idx, X, j, cent[j, :]]
            cent[j, :] = dba_centroid(centroid_data)

        # Calculate distances (simplified without multiprocessing)
        for p in range(m):
            for q in range(k):
                D[p, q] = cDTW([X[p, :], cent[q, :], len(X[p, :])])

        idx = D.argmin(1)
        if np.array_equal(old_idx, idx):
            break

    return idx, cent


def cDTW(data):
    t, r, W = data[0], data[1], data[2]
    N, M = t.shape[0], r.shape[0]
    D = np.ones((N+1, M+1))*np.inf

    D[0, 0,] = 0
    for i in range(1, N+1):
        for j in range(max(1, i-W-1), min(M+1, i+W-1)):
            cost = (t[i-1] - r[j-1])**2
            D[i, j] = cost + min([D[i-1,j],D[i-1,j-1],D[i,j-1]])

    return np.sqrt(D[N, M])


def dba_centroid(data):
    idx, X, j, cur_center = data[0], data[1], data[2], data[3]
    a = []
    for i in range(len(idx)):
        if idx[i] == j:
            opt_x = X[i]
            a.append(opt_x)
    a = np.array(a)

    if len(a) == 0:
        return np.zeros(X.shape[1])

    return DBA(a, cur_center)


def DBA(sequences, cur_center):
    average = cur_center
    return DBA_one_iteration(average, sequences)


def DBA_one_iteration(C, sequences):
    NIL = -1
    DIAGNOAL = 0
    LEFT = 1
    UP = 2
    MAX_SEQ_LENGTH = int(len(C))
    costMatrix = np.zeros((MAX_SEQ_LENGTH, MAX_SEQ_LENGTH))
    pathMatrix = np.ones((MAX_SEQ_LENGTH, MAX_SEQ_LENGTH))
    optimalPathLength = np.ones((MAX_SEQ_LENGTH, MAX_SEQ_LENGTH))

    tupleAssociation = [[] for i in range(len(C))]
    res = 0.0
    centerLength = len(C)

    for T in sequences:
        seqLength = len(T)

        costMatrix[0][0] = distanceTo(C[0], T[0])
        pathMatrix[0][0] = NIL
        optimalPathLength[0][0] = 0

        for i in range(1, centerLength):
            costMatrix[i][0] = costMatrix[i-1][0] + distanceTo(C[i], T[0])
            pathMatrix[i][0] = UP
            optimalPathLength[i][0] = i

        for j in range(1, seqLength):
            costMatrix[0][j] = costMatrix[0][j-1] + distanceTo(T[j], C[0])
            pathMatrix[0][j] = LEFT
            optimalPathLength[0][j] = j

        for i in range(1, centerLength):
            for j in range(1, seqLength):
                indiceRes = ArgMin3(costMatrix[i-1][j-1], costMatrix[i][j-1], costMatrix[i-1][j])
                pathMatrix[i][j] = indiceRes
                if indiceRes == DIAGNOAL:
                    res = costMatrix[i-1][j-1]
                    optimalPathLength[i][j] = optimalPathLength[i-1][j-1] + 1
                elif indiceRes == LEFT:
                    res = costMatrix[i][j-1]
                    optimalPathLength[i][j] = optimalPathLength[i][j-1] + 1
                elif indiceRes == UP:
                    res = costMatrix[i-1][j]
                    optimalPathLength[i][j] = optimalPathLength[i-1][j] + 1
                costMatrix[i][j] = res + distanceTo(C[i], T[j])

        nbTuplesAverageSeq = int(optimalPathLength[centerLength-1][seqLength-1] + 1)

        i = centerLength - 1
        j = seqLength - 1

        for t in range(nbTuplesAverageSeq-1, -1, -1):
            tupleAssociation[i].append(T[j])
            if pathMatrix[i][j] == DIAGNOAL:
                i = i - 1
                j = j - 1
            elif pathMatrix[i][j] == LEFT:
                j = j - 1
            elif pathMatrix[i][j] == UP:
                i = i - 1

    for t in range(0, centerLength):
        if len(tupleAssociation[t]) > 0:
            C[t] = np.mean(tupleAssociation[t])

    return C


def ArgMin3(a, b, c):
    if a < b:
        if a < c:
            return 0
        else:
            return 2
    else:
        if b < c:
            return 1
        else:
            return 2


def distanceTo(a, b):
    return (a - b) * (a - b)


class KDBAClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # KDBA doesn't support precomputed distance matrices
        if self.distance_matrix is not None:
            print("Warning: KDBA does not support precomputed distance matrices. Using original data.")
        
        # Extract parameters with defaults
        max_iter = self.params.get('max_iter', 100)
        random_state = self.params.get('random_state', None)
        
        # Use the actual kDBA implementation with parameters
        predictions, _ = kDBA(X, self.n_clusters, max_iter=max_iter, random_state=random_state)
        
        elapsed = time.time() - start_time
        return predictions, elapsed