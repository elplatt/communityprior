import sys
import pandas

in_file = sys.argv[1]
node_head = sys.argv[2]
community_head = sys.argv[3]
try:
    prob_head = sys.argv[4]
except IndexError:
    prob_head = None

print "node_id,community_id,member_prob\n"
df = pandas.DataFrame.from_csv(in_file, index_col=None)
for i, row in df.iterrows():
    node_id = int(row[node_head])
    community_id = int(row[community_head])
    if prob_head:
        member_prob = row[prob_head]
    else:
        member_prob = 1.0
    print "%d,%d,%f\n" % (node_id, community_id, member_prob)
    