import sys

#import dirichlet
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

def get_id_to_index(dict_file):
    id_to_index = {}
    with open(dict_file, "rb") as f_dict:
        f_dict.next()
        for row in f_dict:
            node_index, node_id = row.rstrip().split(",")
            id_to_index[int(node_id)] = int(node_index)
    return id_to_index

def estimate_simple(com_data, id_to_index):
    
    # Count number of nodes and communities
    nodes = set(com_data["node_id"])
    coms = set(com_data["community_id"])
    num_nodes = len(nodes)
    num_coms = len(coms)
    com_id_to_index = {}
    for i, com_id in enumerate(sorted(list(coms))):
        com_id_to_index[com_id] = i
    
    print "Estimating %d nodes and %d communities (simple)" % (num_nodes, num_coms)
    
    # Count size of each community, node
    print "Summing weights over all communities, nodes"
    com_nodecount = [0.0] * num_coms
    node_comcount = [0.0] * num_nodes
    for i, row in com_data.iterrows():
        node_id = int(row['node_id'])
        node_index = id_to_index[node_id]
        com_index = com_id_to_index[int(row['community_id'])]
        node_comcount[node_index] += row['member_prob']
        com_nodecount[com_index] += row['member_prob']
    
    # Estimate by simple averaging
    # The above totals are used to normalize samples as we add them,
    # e.g. if there are three nodes in a community, each will contribute 1/3 when added.
    print "Averaging distributions"
    alpha = np.zeros((num_coms,))
    beta = np.zeros((num_nodes,))
    for i, row in com_data.iterrows():
        if i % 10000 == 0:
            print "  %d/%d" % (i, len(com_data))
        node_index = id_to_index[int(row['node_id'])]
        com_index = com_id_to_index[int(row['community_id'])]
        mem_p = row['member_prob']
        alpha[com_index] += float(mem_p) / float(node_comcount[node_index])
        beta[node_index] += float(mem_p) / float(com_nodecount[com_index])
    
    # Scale using conventional values
    alpha = alpha * 50.0 / float(num_coms)
    beta = beta * 200.0 / float(num_nodes)
    
    return (alpha, beta)

if __name__ == '__main__':
    # Load data
    com_data = pd.DataFrame.from_csv(sys.argv[1], index_col=None)
    id_to_index = get_id_to_index(sys.argv[2])
    
    # Estimate
    alpha, beta = estimate_simple(com_data, id_to_index)
    
    # Write output
    with open(sys.argv[3], "wb") as f:
        f.write("alpha_k\n")
        for alpha_k in alpha:
            f.write("%f\n" % alpha_k)
    with open(sys.argv[4], "wb") as f:
        f.write("beta_v\n")
        for beta_v in beta:
            f.write("%f\n" % beta_v)