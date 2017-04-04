import math
import time
import numpy as np
import pandas as pd

def weighted_overlapping(a, b):
    '''Calculate NMI for two covers with overlapping communities and weighted
    memberships.
    Arguments should be two covers with the following columns:
    - node_id
    - community_id
    - member_prob
    '''
    
    # Translate from ids to indexes
    node_ids = sorted(list(set(a["node_id"]).union(set(b["node_id"]))))
    num_nodes = len(node_ids)
    id_to_index = {}
    for node_index, node_id in enumerate(node_ids):
        id_to_index[node_id] = node_index
    
    # Construct node-community matrix, cells are membership weights
    print "Constructing weight matrices"
    num_coms_a = a["community_id"].max() + 1
    num_coms_b = b["community_id"].max() + 1
    weights_a = np.zeros((num_nodes, num_coms_a))
    weights_b = np.zeros((num_nodes, num_coms_b))
    node_max_a = np.zeros(num_nodes)
    node_max_b = np.zeros(num_nodes)
    for i, row in a.iterrows():
        # Set weight
        node_id = row["node_id"]
        node = id_to_index[node_id]
        com = row["community_id"]
        w = row["member_prob"]
        weights_a[node,com] = w
        # Check node's max weight for normalization
        if w > node_max_a[node]:
            node_max_a[node] = w
    for i, row in b.iterrows():
        # Set weight
        node = id_to_index[row["node_id"]]
        com = row["community_id"]
        w = row["member_prob"]
        weights_b[node,com] = w
        # Check node's max weight for normalization
        if w > node_max_b[node]:
            node_max_b[node] = w
    
    # Normalize weight matrices
    # np.divide(a,b) divides each row of a by the elements of b component-wise
    # We want to all entries for a node (rows) by the same element of b, so transpose
    print "Normalizing weights"
    weights_a = np.divide(weights_a.transpose(), node_max_a).transpose()
    weights_b = np.divide(weights_b.transpose(), node_max_b).transpose()
    
    # Get joint distribution and entropies
    a_b, a_notb, nota_b = get_joint_dist(weights_a, weights_b)
    
    return _wo_from_joint(a_b, a_notb, nota_b)
    
def get_joint_dist(weights_a, weights_b):
    num_nodes, num_coms_a = weights_a.shape
    num_nodes, num_coms_b = weights_b.shape    
    a_b = np.zeros((num_coms_a, num_coms_b))
    a_notb = np.zeros((num_coms_a, num_coms_b))
    nota_b = np.zeros((num_coms_a, num_coms_b))
    # To save memory, we can find nota_notb from normalization
    
    print "Calculating joint distribution"
    total = num_nodes * num_coms_a * num_coms_b
    done = 0
    start = time.time()
    last = start
    one_norm = 1.0 / float(num_nodes)
    try:
        for node in range(num_nodes):
            for com_a in range(num_coms_a):
                w_a = weights_a[node,com_a] / float(num_nodes)
                for com_b in range(num_coms_b):
                    w_b = weights_b[node,com_b] / float(num_nodes)
                    m = min(w_a,w_b)
                    a_b[com_a,com_b] += m
                    a_notb += w_a - m
                    nota_b += w_b - m
                    done += 1
    except KeyboardInterrupt:
        print "%d/%d (%2.4f%%) in %d seconds" % (done, total, 100.0 * done / float(total), time.time() - start)
        raise
    print "Finished calculating joint distribution"

    return (a_b, a_notb, nota_b)

def get_H_joint(a_b, a_notb, nota_b):
    num_coms_a, num_coms_b = a_b.shape
    H_kl = np.zeros((num_coms_a,num_coms_b))
    print "Calculating joint entropy"
    total = float(com_a * com_b)
    done = 1.0
    start = time.time()
    last = start
    print "Caclulating joint entropy"
    for com_a in range(num_coms_a):
        for com_b in range(num_coms_b):
            # Calculate pairwise
            h = -1.0 * a_b[com_a,com_b] * math.log(a_b[com_a,com_b],2)
            h += -1.0 * a_notb[com_a,com_b] * math.log(a_notb[com_a,com_b], 2)
            h += -1.0 * nota_b[com_a,com_b] * math.log(nota_b[com_a,com_b], 2)
            nota_notb = 1.0 - a_b[com_a,com_b] - a_notb[com_a,com_b] - nota_b[com_a,com_b]
            h += -1.0 * nota_notb * math.log(nota_notb, 2)
            H_kl[com_a,com_b] = h
            if time.time() - last >= 60:
                done += 1.0
                print "%d/%d (%2.4f%%)" % (done, total, 100.0 * done / total)
                last = time.time()
    print "Done calculating joint entropy"
    return H_kl

def get_H_marginal(a_b, a_notb, nota_b):
    num_coms_a, num_coms_b = a_b.shape
    H_k = np.zeros(num_coms_a)
    # Calculate marginal entropy for a
    print "Calculating marginal entropy"
    for com_a in range(num_coms_a):
        p_a = a_b[com_a,:].sum() + a_notb[com_a,:].sum()
        H_k[com_a] += -1.0 * p_a * math.log(p_a, 2)
        H_k[com_a] += -1.0 * (1.0 - a) * math.log(1.0 - a, 2)
    return H_k

def _wo_from_joint(a_b, a_notb, nota_b):
    '''LFK B.11'''
    num_coms_a, num_coms_b = a_b.shape
    H_kl = get_H_joint(a_b, a_notb, nota_b)
    H_k = get_H_marginal(a_b, a_notb, nota_b)
    H_l = get_H_marginal(a_b.transpose(), nota_b.transpose(), a_notb.transpose())
        
    # LFK B.11
    print "Calculating normalized conditional entropy"
    H_cond_a = 0.0
    for k in range(num_coms_a):
        H_xk_given_y = 0.0
        # Find conditional entropy for community k (B.9)
        for l in range(num_coms_b):
            H_xk_given_yl = H_kl[k,l] - H_l[l]
            if l == 0 or H_xk_given_yl < H_xk_given_y:
                H_xk_given_y = H_xk_given_yl
        # Normalize and add to total
        H_cond_a += H_xk_given_y / H_k[k]
    # Normalize
    H_cond_a /= num_coms_a
    
    print "Calculating normalized conditional entropy"
    H_cond_b = 0.0
    for l in range(num_coms_b):
        H_yl_given_x = 0.0
        # Find conditional entropy for community k (B.9)
        for k in range(num_coms_a):
            H_yl_given_xk = H_kl[k,l] - H_k[k]
            if l == 0 or H_yl_given_xk < H_yl_given_x:
                H_yl_given_x = H_yl_given_xk
        # Normalize and add to total
        H_cond_b += H_yl_given_x / H_l[l]
    # Normalize
    H_cond_b /= num_coms_b
    
    return 1.0 - (H_cond_a + H_cond_b) / 2.0