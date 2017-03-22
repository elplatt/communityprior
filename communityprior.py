import sys

import dirichlet
import numpy
import pandas

def data_to_matrix(com_data):
    # Count number of nodes and communities
    # Assume the first of each is "0"
    num_nodes = com_data['node_id'].max() + 1
    num_coms = com_data['community_id'].max() + 1
    
    # Convert to matrix
    node_com = numpy.zeros((num_nodes,num_coms))
    for i, row in com_data.iterrows():
        node_com[int(row["node_id"]),int(row["community_id"])] = row["member_prob"]
    
    return node_com

def normalize_rows(m):
    return m / m.sum(axis=1)[:,numpy.newaxis]

def load_as_matrix(filename):
    # node_id, community_id, member_prob
    com_data = pandas.DataFrame.from_csv(filename, index_col=None)
    return data_to_matrix(com_data)

def estimate_simple(p):
    # Estimate by averaging rows
    return p.mean(axis=0)

if __name__ == '__main__':
    # Load
    node_com = load_as_matrix(sys.argv[1])
    
    # Convert rows/cols into probability distributions
    p_com = normalize_rows(node_com)
    p_node = normalize_rows(node_com.transpose())
    
    # Estimate dirichlet parameter for distribution over communities (topics)
    #alpha = dirichlet.mle(p_com)
    alpha = 50.0 * estimate_simple(p_com)
    
    # Estimate dirichlet parameter for distribution over nodes (terms)
    #beta = dirichlet.mle(p_node)
    beta = 200.0 * estimate_simple(p_node)

    # Write output
    with open(sys.argv[2], "wb") as f:
        f.write("alpha_k\n")
        for alpha_k in alpha:
            f.write("%f\n" % alpha_k)
    with open(sys.argv[3], "wb") as f:
        f.write("beta_v\n")
        for beta_v in beta:
            f.write("%f\n" % beta_v)