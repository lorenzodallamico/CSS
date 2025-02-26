import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import bmat, diags


def load_graph(filename, directory, weighted = True, symmetric = True):
    '''This function loads a graph from the folder Data and outputs its sparse adjacency matrix
    
    Use: A = load_graph(filename, directory)
    
    Inputs:
        * filename (str): name of the dataset. Available names are all the files in the Data folder. 
        * directory (str): location of the file
        
    Optional inputs:
        * weighted (bool): if True (default) it returns a  weighted adjacency matrix and an unweighted one otherwise
        * symmetric (bool): it True (default) it forces the adjacency matrix to be symmetric
        
    Output:<
        * A (sparse array): graph adjacency matrix
    '''

    df = pd.read_csv(f'{directory}/{filename}')

    # exctract all nodes that appear in the network
    all_nodes = np.unique(df[['i', 'j']])

    # extract the number of nodes
    n = len(all_nodes)

    # create a mapping between the nodes' identities and integers
    NodeMapper = dict(zip(all_nodes, np.arange(n)))

    # map the nodes
    df.i = df.i.map(lambda x: NodeMapper[x])
    df.j = df.j.map(lambda x: NodeMapper[x])

    # get the adjacency matrix
    A = csr_matrix((df.w, (df.i, df.j)), shape = (n,n))

    if not weighted:
        A = A.sign()
        if symmetric:
            A = (A + A.T).sign()

    if symmetric:
        A = (A + A.T)/2

    return A



def SpectralRadius(A):
    '''This function computes the spectral radius of an Hermitian matrix A
    Use: ρ = SpectralRadius(A)
    
    Input: 
        * A (scipy sparse array)
        
    Output:
        * ρ (float)
        
    '''

    ρ, _ = eigsh(A.astype(float), k = 1, which = 'LM')

    return ρ[0]


def SpectralRadiusNB(A):
    '''This function computes the spectral radius of the non-backtracking matrix
    Use: ρ = SpectralRadiusNB(A)
    
    Input:
        * A (scipy sparse array): graph adjacency matrix
        
    Outoput:
        * ρ (float)
    '''

    n, _ = A.shape
    D = diags(A@np.ones(n))
    Id = diags(np.ones(n))
    Bp = bmat([[A, -Id], [D-Id, None]])

    ρB, _  = eigs(Bp, k = 1, which = 'LM')
    return ρB[0].real