import numpy as np
from scipy.sparse import csr_matrix

def fastER(n, c):
    ''' Function that generates the adjacency matrix A and a networkx graph for the ER model with n nodes and average degree c

    Use: g = fastER(n, c)
    
    Input:
        * n (int) : graph size
        * c (float) : average connectivity of the network
       
    Output:
        * A (scipy sparse matrix): graph adjacency matrix
    '''

    fs = list()
    ss = list()

    # we choose the nodes that should get connected (n+1 comes from the fact that we expect c/2 terms to be so that i = j)
    fs = np.random.choice(n, int((n+1)*c)) 
    ss = np.random.choice(n, int((n+1)*c))
    
    # create the edge list
    edge_list = np.column_stack((fs,ss))

    # remove edges appearing more than once (the fraction of these edges is vanishing)
    edge_list = np.unique(edge_list, axis = 0) 

    # keep only the edges such that A_{ij} = 1 and i > j
    edge_list = edge_list[edge_list[:,0] > edge_list[:,1]] 

    # build and symmetrize the adjacency matrix
    A = csr_matrix((np.ones(len(edge_list[:,0])), (edge_list[:,0], edge_list[:,1])), shape=(n, n))
    A = A + A.transpose()

    return A

