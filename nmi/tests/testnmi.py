import math
import unittest
import numpy as np
import numpy.testing as nptest
import pandas as pd
import nmi

# Hand calculations in ELP-UM-002 p39

df_a = pd.DataFrame.from_dict({
    "node_id": [0, 0, 1],
    "community_id": [0, 1, 1],
    "member_prob": [1.0, 0.75, 1.0]
})

df_b = pd.DataFrame.from_dict({
    "node_id": [0, 0, 1, 1],
    "community_id": [0, 1, 0, 1],
    "member_prob": [0.75, 0.5, 0.25, 0.75]
})

true_weights_a = np.array([
    [1.0, 0.75],
    [0, 1.0]
])

true_weights_b = np.array([
    [0.75, 0.5],
    [0.25, 0.75]
])

true_a_b = np.array([
    [0.375, 0.25],
    [0.5, 0.625]
])

true_a_notb = np.array([
    [0.125, 0.25],
    [0.375, 0.25]
])

true_nota_b = np.array([
    [0.125, 0.375],
    [0, 0]
])

true_a_marginal = np.array([0.5, 0.875])
true_b_marginal = np.array([0.5, 0.625])

true_H_kl = np.array([
    [3.0 - (3.0/4.0)*math.log(3,2),
     5.0/2.0 - (3.0/8.0)*math.log(3,2)],
    [2.0 - (3.0/8.0)*math.log(3,2),
     11.0/4.0 - (5.0/8.0)*math.log(5,2)]
])

true_H_k = np.array([
    1.0,
    3.0 - (7.0/8.0)*math.log(7,2)
])

true_N = 1.0 - (
    0.25*(
        2.0 - 0.75*math.log(3,2)
        +
        ((3.0/8.0)*math.log(3,2) - 0.25)
        / (3.0 - (7.0/8.0)*math.log(7,2))
    )
    +
    0.25*(
        2.0 - 0.75*math.log(3,2)
        +
        ((7.0/8.0)*math.log(7,2) - (5.0/8.0)*math.log(5,2) - 0.25)
        / (3.0 - (5.0/8.0)*math.log(5,2) - (3.0/8.0)*math.log(3,2))
    )
)

# Unweighted

df_a_uw = pd.DataFrame.from_dict({
    "node_id": [0, 1, 2, 0, 1, 2],
    "community_id": [0, 0, 0, 1, 1, 2],
    "member_prob": [1, 1, 1, 1, 1, 1]
})

df_b_uw = pd.DataFrame.from_dict({
    "node_id": [0, 2, 0, 1, 1, 2],
    "community_id": [0, 0, 1, 1, 2, 2],
    "member_prob": [1, 1, 1, 1, 1, 1]
})

class TestWeighted(unittest.TestCase):
    
    def test_weights(self):
        weights_a, weights_b = nmi.get_weights(df_a, df_b, normalize=False)
        nptest.assert_array_equal(weights_a, true_weights_a)
        nptest.assert_array_equal(weights_b, true_weights_b)
    
    def test_joint(self):
        a_b, a_notb, nota_b = nmi.get_joint_dist(true_weights_a, true_weights_b)
        nptest.assert_array_almost_equal(a_b, true_a_b)
        nptest.assert_array_almost_equal(a_notb, true_a_notb)
        nptest.assert_array_almost_equal(nota_b, true_nota_b)
    
    def test_marginals(self):
        a_marginal = nmi.get_marginal(true_weights_a)
        b_marginal = nmi.get_marginal(true_weights_b)
        nptest.assert_array_almost_equal(a_marginal, true_a_marginal)
        nptest.assert_array_almost_equal(b_marginal, true_b_marginal)
        
    def test_H_joint(self):
        H_kl = nmi.get_H_joint(true_a_b, true_a_notb, true_nota_b)
        nptest.assert_array_almost_equal(H_kl, true_H_kl)
    
    def test_H_marginal(self):
        H_k = nmi.get_H_marginal(true_a_marginal)
        nptest.assert_array_almost_equal(H_k, true_H_k)
        
    def test_nmi(self):
        N = nmi._from_joint(
            true_a_b, true_a_notb, true_nota_b,
            true_a_marginal, true_b_marginal)
        self.assertAlmostEqual(N, true_N)
    
    def test_integrated(self):
        N = nmi.weighted_overlapping(df_a, df_b, normalize=False)
        self.assertAlmostEqual(N, true_N)

class TestReduction(unittest.TestCase):
    
    def test_recuction(self):
        N_w = nmi.weighted_overlapping(df_a_uw, df_b_uw)
        N_uw = nmi.unweighted_overlapping(df_a_uw, df_b_uw)
        self.assertAlmostEqual(N_w, N_uw)

if __name__ == '__main__':
    unittest.main()