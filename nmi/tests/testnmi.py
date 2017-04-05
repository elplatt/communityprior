import unittest
import numpy as np
import pandas as pd
import nmi

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

class TestAll(unittest.TestCase):
    
    def test_weights(self):
        weights_a, weights_b = nmi.get_weights(df_a, df_b, normalize=False)
        self.assertEqual(weights_a, true_weights_a)
        self.assertEqual(weights_b, true_weights_b)
        
if __name__ == '__main__':
    unittest.main()