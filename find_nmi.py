import sys
import pandas as pd
import nmi

test_file = sys.argv[1]
truth_file = sys.argv[2]

df_test = pd.DataFrame.from_csv(test_file, index_col=None)
df_truth = pd.DataFrame.from_csv(truth_file, index_col=None)

res = nmi.weighted_overlapping(df_test, df_truth)

print "NMI: ", res
