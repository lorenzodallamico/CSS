import numpy as np


def propagateMajorityCascade(A, failure_prob, alpha):
    ''''
    Use: not_failed = propagateMajorityCascade(A, failure_prob, alpha)

    Inputs:
        * A (scipy sparse array): graph adjacency matrix
        * failure_prob (float): initial failure probability
        * alpha (float): fraction of failed neighbors needed to cause the cascade effect

    Output:
        * not_failed (list): nodes that did not fail
    '''

    n, _ = A.shape
    d = A@np.ones(n)

    # select the failures
    Nfails = np.random.binomial(n, failure_prob)
    failed = np.random.choice(np.arange(n), Nfails, replace = False)
    
    # status = 1 means that node failed
    status = np.zeros(n)
    status[failed] = 1

    flag = 0

    while flag == 0:
        
        # condition that triggers the cascading failure
        # idx = (1 - status) * (A@status) > alpha * d
        idx = (1-status) * (A@status) * (1+alpha)/alpha > d
        status[idx] = 1

        if np.sum(idx) == 0:
            flag = 1
                
    return np.where(status == 0)[0]


def jaccard_iterators(a,b):
    '''Compute the Jaccard similarity between two sets defined by iterators'''

    A, B = set(a), set(b)

    return len(A.intersection(B))/len(A.union(B))

def RunCascade(A, f, n_iter):
    '''This function runs a cascade according to a majority rule modeled by the vector f
    
    Use: S = RunCascade(A, f)
    
    Inputs:
        * A (scipy sparse array): graph adjacency matrix
        * f (array): vector with node threshold values. Every node will propagate its state to 1 if a fraction f of it neighbors is in class 1
        * n_iter (int): number of iterations

    Outputs:
        * S (array): fraction of nodes in class 1 at the across the iterations
    '''

    n, _ = A.shape
    d = A@np.ones(n)

    # choose the random seed and initialize the state
    seed = np.random.choice(n)
    state = np.zeros(n)
    state[seed] = 1
    S = []

    for i in range(n_iter):
        # update the variable state
        idx = (1 - state) * (A@state) > f * d
        state[idx] = 1 
        S.append(np.mean(state))
     
    return np.array(S)