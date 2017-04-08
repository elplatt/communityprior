import sys
import pandas as pd
import nmi

test_file = sys.argv[1]
truth_file = sys.argv[2]
try:
    threshold = float(sys.argv[3])
except IndexError:
    threshold = 1.0
df_test = pd.DataFrame.from_csv(test_file, index_col=None)
df_truth = pd.DataFrame.from_csv(truth_file, index_col=None)

res = nmi.unweighted_overlapping(df_test, df_truth, threshold)

print "NMI: ", res
