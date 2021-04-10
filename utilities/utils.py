import numpy as np
import itertools as it
from scipy.spatial.distance import euclidean as measure


def get_nodes(landmark):
    return np.array([ [pnt.x, pnt.y] for pnt in landmark])

def normalize_nodes(nodes, scaler):
	rescaled_nodes = nodes * scaler
	return rescaled_nodes.astype('int32')


def make_adjacency_matrix(nodes):
	nb_node = len(nodes)  
	pairwise = list(it.product(nodes, nodes))	
	weighted_edges = np.array([ measure(*item) for item in pairwise ])
	return np.reshape(weighted_edges, (nb_node, nb_node)) / np.max(weighted_edges)