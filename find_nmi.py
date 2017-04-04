import sys
import pandas as pd
import nmi

test_file = sys.argv[1]
truth_file = sys.argv[2]

df_test = pd.DataFrame.from_csv(test_file)
df_truth = pd.DataFrame.from_csv(truth_file)

res = nmi.weighted_overlapping(df_test, df_truth)

print "NMI: ", res
