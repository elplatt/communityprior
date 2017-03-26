import sys

com_file = sys.argv[1]

communities = set()
nodes = set()

with open(com_file) as f_com:
    f_com.next()
    for row in f_com:
        node_id, community_id, member_prob = row.rstrip().split(",")
        communities.add(int(community_id))
        nodes.add(int(node_id))

print "%d communities, %d - %d" % (len(communities), min(list(communities)), max(list(communities)))
print "%d nodes, %d - %d" % (len(nodes), min(list(nodes)), max(list(nodes)))
