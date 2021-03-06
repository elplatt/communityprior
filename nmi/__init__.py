import math
import sys
import time
import numpy as np
import pandas as pd
import scipy.stats as spstats

def unweighted_overlapping(a, b, threshold=1.0, normalize=False):
    '''Calculate NMI for two covers with overlapping communities and unweighted
    memberships.
    Arguments should be two covers with the following columns:
    - node_id
    - community_id
    - member_prob (1 or 0)
    '''
    member_a, member_b, num_nodes = get_membership(a, b, threshold, normalize)
    a_marginal = get_unweighted_marginal(member_a, num_nodes)
    b_marginal = get_unweighted_marginal(member_b, num_nodes)
    a_b, a_notb, nota_b = get_unweighted_joint_dist(member_a, member_b, num_nodes)
    return _from_joint(a_b, a_notb, nota_b, a_marginal, b_marginal)

def get_membership(a, b, threshold=1.0, normalize=False):
    # Translate nodes from ids to indexes
    node_ids = sorted(list(set(a["node_id"]).union(set(b["node_id"]))))
    num_nodes = len(node_ids)
    coms_ids = set(a["community_id"]).union(set(b["community_id"]))
    id_to_index = {}
    for node_index, node_id in enumerate(node_ids):
        id_to_index[node_id] = node_index

    # Get maximum weights for each community
    if normalize:
        com_max_a = dict([(x,0.0) for x in com_ids])
        com_max_b = dict([(x,0.0) for x in com_ids])
        for i, row in a.iterrows():
            com_id = row["community_id"]
            w = row["member_prob"]
            if w > com_max_a[com_id]:
                com_max_a[com_id] = w
        for i, row in b.iterrows():
            com_id = row["community_id"]
            w = row["member_prob"]
            if w > com_max_b[com_id]:
                com_max_b[com_id] = w
    
    # Construct set of nodes indexes belonging to each community id
    print "Constructing membership sets"
    com_a_ids = set(a["community_id"])
    com_b_ids = set(b["community_id"])
    node_count_a = dict([(com_id, 0) for com_id in com_a_ids])
    node_count_b = dict([(com_id, 0) for com_id in com_b_ids])
    member_a_by_id = dict([(com_id, set()) for com_id in com_a_ids])
    member_b_by_id = dict([(com_id, set()) for com_id in com_b_ids])
    for i, row in a.iterrows():
        # Set weight
        node_id = row["node_id"]
        node = id_to_index[node_id]
        com_id = int(row["community_id"])
        w = float(row["member_prob"])
        if normalize and com_max_a[com_id] > 0:
            w /= com_max_a[com_id]
        if w >= threshold:
            member_a_by_id[com_id].add(node)
    for i, row in b.iterrows():
        # Set weight
        node = id_to_index[row["node_id"]]
        com_id = int(row["community_id"])
        w = float(row["member_prob"])
        if normalize and com_max_b[com_id] > 0:
            w /= node_max_b[com_id]
        if w >= threshold:
            member_b_by_id[com_id].add(node)

    # Removes communities with < 1 nodes
    print "Removing communities with < 1 nodes or all nodes"
    for com_id in list(com_a_ids):
        size = len(member_a_by_id[com_id])
        if size < 1 or size == num_nodes:
            com_a_ids.remove(com_id)
    for com_id in list(com_b_ids):
        size = len(member_b_by_id[com_id])
        if size < 1 or size == num_nodes:
            com_b_ids.remove(com_id)

    # Represent each community as a set of member nodes
    print "Converting list of sets"
    com_a_ids = sorted(list(com_a_ids))
    com_b_ids = sorted(list(com_b_ids))
    com_a_id_to_index = {}
    com_b_id_to_index = {}
    member_a = []
    member_b = []
    for com_index, com_id in enumerate(com_a_ids):
        member_a.append(member_a_by_id[com_id])
        com_a_id_to_index[com_id] = com_index
    for com_index, com_id in enumerate(com_b_ids):
        member_b.append(member_b_by_id[com_id])
        com_b_id_to_index[com_id] = com_index
    print "%d nodes, (%d, %d) communities" % (num_nodes, len(member_a), len(member_b))
    
    return (member_a, member_b, num_nodes)

def get_unweighted_joint_dist(member_a, member_b, num_nodes):
    num_coms_a = len(member_a)
    num_coms_b = len(member_b)
    a_b = np.zeros((num_coms_a, num_coms_b))
    a_notb = np.zeros((num_coms_a, num_coms_b))
    nota_b = np.zeros((num_coms_a, num_coms_b))
    # To save memory, we can find nota_notb from normalization
    
    print "Calculating joint distribution"
    total = num_coms_a * num_coms_b
    done = 0
    start = time.time()
    last = start
    try:
        for com_a, nodes_a in enumerate(member_a):
            for com_b, nodes_b in enumerate(member_b):
                a_b[com_a,com_b] = len(nodes_a.intersection(nodes_b))
                a_notb[com_a,com_b] = len(nodes_a.difference(nodes_b))
                nota_b[com_a,com_b] = len(nodes_b.difference(nodes_a))
                done += 1
            t = time.time()
            if t - last > 60:
                print "%d/%d (%2.4f%%) in %d seconds" % (done, total, 100.0 * done / float(total), time.time() - start)
                last = t
    except KeyboardInterrupt:
        print "%d/%d (%2.4f%%) in %d seconds" % (done, total, 100.0 * done / float(total), time.time() - start)
        raise
    a_b = a_b / float(num_nodes)
    a_notb = a_notb / float(num_nodes)
    nota_b = nota_b / float(num_nodes)
    print "Finished calculating joint distribution"
    return (a_b, a_notb, nota_b)

def get_unweighted_marginal(member, num_nodes):
    p = np.array([len(x) for x in member]) / float(num_nodes)
    for i in range(len(p)):
        if p[i] == 0.0:
            print "Index %d p=0" % i
        if p[i] == 1.0:
            print "Index %d p=1" % i
    return p

def weighted_overlapping(a, b, normalize=True):
    '''Calculate NMI for two covers with overlapping communities and weighted
    memberships.
    Arguments should be two covers with the following columns:
    - node_id
    - community_id
    - member_prob
    '''
    member_a, weights_a, member_b, weights_b, num_nodes = get_weights(a, b, normalize)
    a_marginal = get_marginal(member_a, weights_a, num_nodes)
    b_marginal = get_marginal(member_b, weights_b, num_nodes)
    a_b, a_notb, nota_b = get_joint_dist(member_a, weights_a, member_b, weights_b, num_nodes)
    return _from_joint(a_b, a_notb, nota_b, a_marginal, b_marginal)

def get_weights(a, b, normalize):
    # Translate from ids to indexes
    node_ids = sorted(list(set(a["node_id"]).union(set(b["node_id"]))))
    num_nodes = len(node_ids)
    id_to_index = {}
    for node_index, node_id in enumerate(node_ids):
        id_to_index[node_id] = node_index

    com_a_ids = set(a["community_id"])
    com_b_ids = set(b["community_id"])
    com_ids = com_a_ids.union(com_b_ids)
    num_coms_a = len(com_a_ids)
    num_coms_b = len(com_b_ids)
    
    # Get maximum weights for each node
    if normalize:
        print "Normalizing weights"
        sys.stdout.flush()
        com_max_a = dict([(x,0.0) for x in com_ids])
        com_max_b = dict([(x,0.0) for x in com_ids])
        for i, row in a.iterrows():
            com_id = row["community_id"]
            w = row["member_prob"]
            if w > com_max_a[com_id]:
                com_max_a[com_id] = w
        for i, row in b.iterrows():
            com_id = row["community_id"]
            w = row["member_prob"]
            if w > com_max_b[com_id]:
                com_max_b[com_id] = w
    
    # Construct weight data structures
    # member_x is list of node sets
    # weight_x is {(com, node) -> weight}
    print "Constructing weight matrices"
    sys.stdout.flush()
    member_a_by_id = dict([(com_id,set()) for com_id in com_a_ids])
    member_b_by_id = dict([(com_id,set()) for com_id in com_b_ids])
    weights_a_by_id = {}
    weights_b_by_id = {}
    for i, row in a.iterrows():
        # Set weight
        node_id = row["node_id"]
        node = id_to_index[node_id]
        com_id = int(row["community_id"])
        w = row["member_prob"]
        if normalize:
            w /= com_max_a[com_id]
        member_a_by_id[com_id].add(node)
        weights_a_by_id[(node,com_id)] = w
    for i, row in b.iterrows():
        # Set weight
        node = id_to_index[row["node_id"]]
        com_id = int(row["community_id"])
        w = row["member_prob"]
        if normalize:
            w /= com_max_b[com_id]
        member_b_by_id[com_id].add(node)
        weights_b_by_id[(node,com_id)] = w
    
    # Remove communities with < 1 nodes or all nodes
    # Find indexes of communities to remove
    print "Removing communities with no/all nodes"
    sys.stdout.flush()
    remove_a = set()
    remove_b = set()
    for com_id in com_a_ids:
        s = sum([weights_a_by_id[(node,com_id)] for node in member_a_by_id[com_id]])
        if s == num_nodes:
            remove_a.add(com_id)
        if len(member_a_by_id[com_id]) < 1:
            remove_a.add(com_id)
    for com_id in com_b_ids:
        s = sum([weights_b_by_id[(node,com_id)] for node in member_b_by_id[com_id]])
        if s == num_nodes:
            remove_b.add(com_id)
        if len(member_b_by_id[com_id]) < 1:
            remove_b.add(com_id)
    com_a_ids = com_a_ids - remove_a
    com_b_ids = com_b_ids - remove_b
    num_coms_a = len(com_a_ids)
    num_coms_b = len(com_b_ids)
    
    # Represent each community as a set of member nodes
    print "Converting list of sets"
    sys.stdout.flush()
    com_a_ids = sorted(list(com_a_ids))
    com_b_ids = sorted(list(com_b_ids))
    com_a_id_to_index = {}
    com_b_id_to_index = {}
    member_a = []
    member_b = []
    weights_a = {}
    weights_b = {}
    for com_index, com_id in enumerate(com_a_ids):
        nodes = member_a_by_id[com_id]
        member_a.append(nodes)
        com_a_id_to_index[com_id] = com_index
        for node in nodes:
            weights_a[(node,com_index)] = weights_a_by_id[(node,com_id)]
    for com_index, com_id in enumerate(com_b_ids):
        nodes = member_b_by_id[com_id]
        member_b.append(nodes)
        com_b_id_to_index[com_id] = com_index
        for node in nodes:
            w = weights_b_by_id[(node,com_id)]
            weights_b[(node,com_index)] = w
    print "%d nodes, (%d, %d) communities" % (num_nodes, len(member_a), len(member_b))
    sys.stdout.flush()

    return (member_a, weights_a, member_b, weights_b, num_nodes)
    
def get_joint_dist(member_a, weights_a, member_b, weights_b, num_nodes):
    num_coms_a = len(member_a)
    num_coms_b = len(member_b)
    a_b = np.zeros((num_coms_a, num_coms_b))
    a_notb = np.zeros((num_coms_a, num_coms_b))
    nota_b = np.zeros((num_coms_a, num_coms_b))
    # To save memory, we can find nota_notb from normalization
    
    print "Calculating joint distribution"
    sys.stdout.flush()
    total = num_coms_a * num_coms_b
    done = 0
    start = time.time()
    last = start
    one_norm = 1.0 / float(num_nodes)
    try:
        for com_a, nodes_a in enumerate(member_a):
            for com_b, nodes_b in enumerate(member_b):
                # These set operations determine which weights we need to check
                both = nodes_a.intersection(nodes_b)
                a_less_b = nodes_a - nodes_b
                b_less_a = nodes_b - nodes_a
                # Calculate joint distribution for this pair of communities
                tot_a_b = 0.0
                tot_a_notb = 0.0
                tot_nota_b = 0.0
                # Update totals based on intersection
                for node in both:
                    wa = weights_a[(node,com_a)]
                    wb = weights_b[(node,com_b)]
                    m = min(wa, wb)
                    tot_a_b += m
                    tot_a_notb += wa - m
                    tot_nota_b += wb - m
                # Update totals based on set differences
                for node in a_less_b:
                    wa = weights_a[(node,com_a)]
                    tot_a_notb += wa
                for node in b_less_a:
                    wb = weights_b[(node,com_b)]
                    tot_nota_b += wb
                # Save results in 2d array
                a_b[com_a,com_b] = tot_a_b / float(num_nodes)
                a_notb[com_a,com_b] = tot_a_notb / float(num_nodes)
                nota_b[com_a,com_b] = tot_nota_b / float(num_nodes)
                done += 1
            t = time.time()
            if t - last > 60:
                print "%d/%d (%2.4f%%) in %d seconds" % (done, total, 100.0 * done / float(total), time.time() - start)
                sys.stdout.flush()
                last = t
    except KeyboardInterrupt:
        print "%d/%d (%2.4f%%) in %d seconds" % (done, total, 100.0 * done / float(total), time.time() - start)
        raise
    print "Finished calculating joint distribution"

    return (a_b, a_notb, nota_b)

def get_marginal(member, weights, num_nodes):
    p = [
        sum([weights[(node,com)] for node in nodes]) / float(num_nodes)
        for com, nodes in enumerate(member)]
    return np.array(p)

def get_H_joint(a_b, a_notb, nota_b):
    num_coms_a, num_coms_b = a_b.shape
    H_kl = np.zeros((num_coms_a,num_coms_b))
    print "Calculating joint entropy"
    total = float(num_coms_a * num_coms_b)
    done = 1.0
    start = time.time()
    last = start
    print "Caclulating joint entropy"
    for com_a in range(num_coms_a):
        for com_b in range(num_coms_b):
            # Calculate pairwise
            if a_b[com_a,com_b] > 0.0:
                h = -1.0 * a_b[com_a,com_b] * math.log(a_b[com_a,com_b],2)
            else:
                h = 0.0
            if a_notb[com_a,com_b] > 0.0:
                h += -1.0 * a_notb[com_a,com_b] * math.log(a_notb[com_a,com_b], 2)
            if nota_b[com_a,com_b] > 0.0:
                h += -1.0 * nota_b[com_a,com_b] * math.log(nota_b[com_a,com_b], 2)
            nota_notb = 1.0 - a_b[com_a,com_b] - a_notb[com_a,com_b] - nota_b[com_a,com_b]
            if nota_notb > 0.0:
                h += -1.0 * nota_notb * math.log(nota_notb, 2)
            H_kl[com_a,com_b] = h
            if time.time() - last >= 60:
                done += 1.0
                print "%d/%d (%2.4f%%)" % (done, total, 100.0 * done / total)
                last = time.time()
    print "Done calculating joint entropy"
    return H_kl

def get_H_marginal(p):
    '''Argument is array of P(node in community k).'''
    # Create an array where each column is a distribution
    antip = 1.0 - p
    pp = np.concatenate((p[:,np.newaxis],antip[:,np.newaxis]),axis=1)
    # Now we can pass the array to scipy to get the entropy
    H = spstats.entropy(pp.transpose(), base=2)
    return H

def _from_joint(a_b, a_notb, nota_b, a_marginal, b_marginal):
    '''LFK B.11'''
    num_coms_a, num_coms_b = a_b.shape
    H_kl = get_H_joint(a_b, a_notb, nota_b)
    H_k = get_H_marginal(a_marginal)
    H_l = get_H_marginal(b_marginal)
        
    # LFK B.11
    print "Calculating normalized conditional entropy"
    H_cond_a = 0.0
    for k in range(num_coms_a):
        H_xk_given_y = 0.0
        # Find conditional entropy for community k (B.9)
        empty = True
        for l in range(num_coms_b):
            # Constraint B.14
            p11 = a_b[k,l]
            p10 = a_notb[k,l]
            p01 = nota_b[k,l]
            p00 = 1.0 - p11 - p10 - p01
            h11 = 0.0
            h00 = 0.0
            h10 = 0.0
            h01 = 0.0
            if p11 > 0:
                h11 -= p11*math.log(p11,2)
            if p11 < 1:
                h11 -= (1.0-p11)*math.log(1.0-p11,2)
            if p00 > 0:
                h00 -= p00*math.log(p00,2)
            if p00 < 1:
                h00 -= (1.0-p00)*math.log(1.0-p00,2)
            if p10 > 0:
                h10 -= p10*math.log(p10,2)
            if p10 < 1:
                h10 -= (1.0-p10)*math.log(1.0-p10,2)
            if p01 > 0:
                h01 -= p01*math.log(p01,2)
            if p01 < 1:
                h01 -= (1.0-p01)*math.log(1.0-p01,2)
            if h11 + h00 < h01 + h10:
                # Constraint not satisfied
                continue
            # Passed constraint, compare to current min
            H_xk_given_yl = H_kl[k,l] - H_l[l]
            if empty or H_xk_given_yl < H_xk_given_y:
                #print "For k=%d, using %d" % (k, l)
                H_xk_given_y = H_xk_given_yl
                empty = False
        # Normalize and add to total
        H_cond_a += H_xk_given_y / H_k[k]
    # Normalize
    H_cond_a /= num_coms_a
    
    print "Calculating normalized conditional entropy"
    H_cond_b = 0.0
    for l in range(num_coms_b):
        H_yl_given_x = 0.0
        # Find conditional entropy for community k (B.9)
        empty = True
        for k in range(num_coms_a):
            # Constraint B.14
            p11 = a_b[k,l]
            p10 = a_notb[k,l]
            p01 = nota_b[k,l]
            p00 = 1.0 - p11 - p10 - p01
            h11 = 0.0
            h00 = 0.0
            h10 = 0.0
            h01 = 0.0
            if p11 > 0:
                h11 -= p11*math.log(p11,2)
            if p11 < 1:
                h11 -= (1.0-p11)*math.log(1.0-p11,2)
            if p00 > 0:
                h00 -= p00*math.log(p00,2)
            if p00 < 1:
                h00 -= (1.0-p00)*math.log(1.0-p00,2)
            if p10 > 0:
                h10 -= p10*math.log(p10,2)
            if p10 < 1:
                h10 -= (1.0-p10)*math.log(1.0-p10,2)
            if p01 > 0:
                h01 -= p01*math.log(p01,2)
            if p01 < 1:
                h01 -= (1.0-p01)*math.log(1.0-p01,2)
            if h11 + h00 < h01 + h10:
                # Constraint not satisfied
                continue
            # Passed constraint, compare to current min
            H_yl_given_xk = H_kl[k,l] - H_k[k]
            if empty or H_yl_given_xk < H_yl_given_x:
                #print "For l=%d, using %d" % (l, k)
                H_yl_given_x = H_yl_given_xk
                empty = False
        # Normalize and add to total
        H_cond_b += H_yl_given_x / H_l[l]
    # Normalize
    H_cond_b /= num_coms_b
    
    return 1.0 - (H_cond_a + H_cond_b) / 2.0