import sys

import dirichlet
import numpy as np
import pandas as pd

def data_to_matrix(com_data):
    # Count number of nodes and communities
    # Assume the first of each is "0"
    num_nodes = com_data['node_id'].max() + 1
    num_coms = com_data['community_id'].max() + 1
    
    # Convert to matrix
    node_com = np.zeros((num_nodes,num_coms))
    for i, row in com_data.iterrows():
        node_com[int(row["node_id"]),int(row["community_id"])] = row["member_prob"]
    
    return node_com

def normalize_rows(m):
    return m / m.sum(axis=1)[:,np.newaxis]

def load_as_matrix(filename):
    # node_id, community_id, member_prob
    com_data = pd.DataFrame.from_csv(filename, index_col=None)
    return data_to_matrix(com_data)

def estimate_mle(com_data):
    # Convert data into matrix
    node_com = data_to_matrix(com_data)
    
    # Convert rows/cols into probability distributions
    p_com = normalize_rows(node_com)
    p_node = normalize_rows(node_com.transpose())
    
    # Estimate dirichlet parameter for distribution over communities (topics)
    alpha = dirichlet.mle(p_com)
    
    # Estimate dirichlet parameter for distribution over nodes (terms)
    beta = dirichlet.mle(p_node)
    
    return (alpha, beta)

def estimate_simple(com_data):
    
    # Count number of nodes and communities
    min_node = int(com_data["node_id"].min())
    min_com = int(com_data["community_id"].min())
    num_nodes = com_data['node_id'].max() - min_node + 1
    num_coms = com_data['community_id'].max() - min_com + 1
    
    print "Estimating %d nodes and %d communities" % (num_nodes, num_coms)
    
    # Count size of each community, node
    print "Summing weights over all communities, nodes"
    com_nodecount = [0.0] * num_coms
    node_comcount = [0.0] * num_nodes
    for i, row in com_data.iterrows():
        node_id = int(row['node_id']) - min_node
        com_id = int(row['community_id']) - min_node
        node_comcount[node_id] += row['member_prob']
        com_nodecount[node_id] += row['member_prob']
    
    # Estimate by simple averaging
    # The above totals are used to normalize samples as we add them,
    # e.g. if there are three nodes in a community, each will contribute 1/3 when added.
    print "Averaging distributions"
    alpha = np.zeros((num_coms,))
    beta = np.zeros((num_nodes,))
    for i, row in com_data.iterrows():
        if i % 1000 == 0:
            print "  %d/%d" % (i, len(com_data))
        node_id = int(row['node_id']) - min_node
        com_id = int(row['community_id']) - min_node
        mem_p = row['member_prob']
        alpha[com_id] += mem_p / com_nodecount[com_id]
        beta[node_id] += mem_p / node_comcount[node_id]
    
    # Scale using conventional values
    #alpha = 50.0 * alpha
    #beta = 200.0 * beta
    
    return (alpha, beta)

if __name__ == '__main__':
    # Load data
    com_data = pd.DataFrame.from_csv(sys.argv[1], index_col=None)
    
    # Estimate
    alpha, beta = estimate_simple(com_data)
    
    # Write output
    with open(sys.argv[2], "wb") as f:
        f.write("alpha_k\n")
        for alpha_k in alpha:
            f.write("%f\n" % alpha_k)
    with open(sys.argv[3], "wb") as f:
        f.write("beta_v\n")
        for beta_v in beta:
            f.write("%f\n" % beta_v)