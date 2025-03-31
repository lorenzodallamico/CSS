import numpy as np
import pandas as pd 
from scipy.sparse import csr_matrix, diags
from copy import copy


def ComputeReachabilityVector(df, T):
    '''Given a temporal network in the form ijt contained in the dataframe df, this function computes the reachability of time-respecting walks of length T
    
    Use: reachability = ComputeReachabilityVector(df, T)
    
    Inputs:
        * df (pandas dataframe): temporal network in the form ijt
        * T (int): number of steps in the walk
        
    Outputs:
        * reachability (array): returns the T reachability values for each time in [1,T]
    '''

    # create a mapping between the nodes and integers
    all_nodes = np.unique(df[['i', 'j']])
    n = len(all_nodes)
    
    # shift the smallest time-stamp to 0
    df.t = df.t - df.t.min()

    # split the edge list according to the time stamp
    dft = [df[df.t == t] for t in range(T)]
    
    # create a list of snapshot adjacency matrices in which each node is connected to itself
    At = [diags(np.ones(n)) for t in range(T)]

    # compute the snapshot adjacency matrices
    for t in range(T):
        if len(dft[t]) > 0:
            A = csr_matrix((np.ones(len(dft[t])), (dft[t].i, dft[t].j)), shape = (n,n))
            At[t] += A + A.T
        
    # compute the reachability vector given the adjacency matrices list
    reachability = ReachabilityFromAdjacencyList(At)
    
    return reachability


def ReachabilityFromAdjacencyList(At):
    '''This function computes the reachability of time respecting paths given a sequence of adjacency matrices with self loops
    
    Use: reachability = ReachabilityFromAdjacencyList(At)
    
    Inputs:
        * At (list of sparse arrays): the entries of this list are the adjacency matrices of each snapshot
        
    Outputs:
        * reachability (array): returns the T reachability values for each time in [1,T], where T is the number of time frames
    
    '''
    
    # compute the reachability matrix R_{ij}(t) = 1  if j can be reached from i in t steps or fewer.
    R = [At[0]]
    T = len(At)
    for t in range(1, T):
        print(f'Progress: {int(100*t/T)}%', end = '\r')
        R.append((At[t]@R[-1]).sign())
        
    # compute the average of the reachability matrix
    reachability = np.array([r.mean() for r in R])
    
    return reachability


def ComputeIntereventStatistic(df_):
    '''Given a contact network in the format ijtτ, this function computes the time elapsed between the end of an 
    interaction between two nodes (ij) and the beginning of a new one between the same nodes. 
    This is repeated for all nodes
    
    Use: intervals = ComputeIntereventStatistic(df)
    
    Inputs:
        * df (pandas dataframe): temporal network in the format ijtτ
        
    Outpus:
        * intervals (array): list of interevent values for all pairs'''
    
    # index the dataframe by the contact indeces
    df = copy(df_)
    df.set_index(['i', 'j'], inplace = True)
    all_pairs = list(df.index)
    intervals = []

    # for each pair, select only the entries of df involving that pair
    for i, pair in enumerate(all_pairs):
        print(f'Progress: {int(i/len(all_pairs)*100)}%', end = '\r')
        ddf = df.loc[pair]
        
        # compute and store the interevent duration
        if len(ddf) > 1:
            intervals.append((ddf.t + ddf.τ - np.roll(ddf.t, 1))[1:].values)
   
    return np.concatenate(intervals)


def ActivityDriven(n, x, η, m, T, verbose = True):
    '''Generates an instance of the activty driven model
    
    Use: dfttau, el = ActivityDriven(n, x, η, m, T)
    
    Inputs:
        * n (int): number of nodes
        * x (array): activity potentials
        * η (float): activity parameter
        * m (int): number of edges per node
        * T (int): number of snapshots

    Optional inputs:
        *verbose (bool): is True (default) is prints the progress

    Output:
        * dfttau (pandas DataFrame): temporal graph in the format i, j, t, τ
        * el (dictionary): The keys are edge (i,j) with i < j and the entries are lists of lists in the format [t_i, t_f], where t_i denotes the beginning of an interaction and t_f the end. Note that t_f - t_i >= 1. 
    '''

    # initialize the dictionary containing the contact timelines of each edge. 
    el = dict()

    for t in range(T):
        if verbose:
            print(f'{int((t+1)*100/T)}%', end = '\r')

        # choose which nodes are active at a particular time-step
        active = np.where(np.random.binomial(1, η*x) == 1)[0]

        for u in active:
            # select the neighbors of the active nodes and add the contacts
            # choose the neighbor among all nodes except u
            neighbors = np.random.choice(list(set(np.arange(n)) - set([u])), m, replace = False)
            for v in neighbors:
                el = _add_contact(el, u, v, t)

    # convert to the ijttau format
    el_ = []

    for e in el.keys():
        i, j = e
        for tt in el[e]:
            el_.append([i, j, tt[0], tt[1]-tt[0]])

    # construct the dataframe in the form ijttau
    dfttau = pd.DataFrame(el_, columns = ['i', 'j', 't', 'τ'])

    return dfttau, el


def _add_contact(el, u, v, t):
    '''This function adds a contact to the dictionary el'''

    # write the edge (u,v) so that u < v
    e = tuple([np.min([u, v]), np.max([u, v])])

    
    if e in el.keys():
        # if the contact is ongoing, increase its duration
        if el[e][-1][1] == t:
            el[e][-1][1] = t+1
        else:
            # if the contact is not ongoing create a new one
            el[e].append([t, t+1])
    
    else:
        # if e is not in the dictionary, add it and set the contact duration to 1
        el[e] = [[t, t+1]]

    return el


def getInterEvent(el):
    '''This function computes the inter-event statistics from the dictionary el'''

    ied = []

    for e in el.keys():
        
        if len(el[e]) > 1:
            for i in range(len(el[e])-1):
                ied.append(el[e][i+1][0] - el[e][i][1])

    return np.array(ied)


def ActractivenessModel(n, L, a, rv, step, r, T, verbose = True):
    '''This function generates an instance of the attractiveness model

    Use: dft, el = ActractivenessModel(n, L, a, rv, step, r, T, verbose = True)
    
    Inputs:
        * n (int): number of nodes
        * L (float): box size
        * a (array): attractivity parameters
        * rv (array): activity vector
        * step (float): length of a step in the movement
        * r (float): radius that defines proximity
        * T (int): number of time-steps

    Optional inputs:
        * verbose (bool): if True (default) the function prints the the progress

    Outputs:
        * dfttau (pandas DataFrame): temporal graph in the format i, j, t, τ
        * el (dictionary): The keys are edge (i,j) with i < j and the entries are lists of lists in the format [t_i, t_f], where t_i denotes the beginning of an interaction and t_f the end. Note that t_f - t_i >= 1. 
    '''

    # initialize the positions of the agents
    xpos = np.random.uniform(0, L, n)
    ypos = np.random.uniform(0, L, n)

    # initialize the dictionary containing the contact timelines of each edge. The keys are edge (i,j) with i < j and the entries are lists of lists in the format
    # [t_i, t_f], where t_i denotes the beginning of an interaction and t_f the end. Note that t_f - t_i >= 1.
    el = dict()

    for t in range(T):
        if verbose:
            print(f'{int((t+1)/T*100)}%', end = '\r')

        # probability of not moving
        p_not_move = np.zeros(n)

        # identify active nodes
        active_nodes = np.where(np.random.binomial(1, rv) == 1)[0]


        for i in active_nodes:
            for j in active_nodes:
                if i < j:
                    # check the distance
                    if (xpos[i] - xpos[j])**2 + (ypos[i] - ypos[j])**2 < r**2:
                        
                        # add the temporal edge
                        el = _add_contact(el, i, j, t)

                        # update the probability of moving 
                        p_not_move[i] = np.max([p_not_move[i], a[j]])
                        p_not_move[j] = np.max([p_not_move[j], a[i]])

        # move the agents
        move = np.where(np.random.binomial(1, 1-p_not_move) == 1)[0]
        theta = np.random.uniform(0, 2*np.pi, len(move))
        xpos[move] += step*np.cos(theta)
        ypos[move] += step*np.sin(theta)

        # apply PBC
        xpos[xpos > L] -= L
        xpos[xpos < 0] += L
        ypos[ypos > L] -= L
        ypos[ypos < 0] += L

    # convert to the ijttau format
    el_ = []

    for e in el.keys():
        i, j = e
        for tt in el[e]:
            el_.append([i, j, tt[0], tt[1]-tt[0]])

    # construct the dataframe in the form ijttau
    dfttau = pd.DataFrame(el_, columns = ['i', 'j', 't', 'τ'])

    return dfttau, el

        