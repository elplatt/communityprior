edge_file = "data/networks/lfr_network_mu0.8.csv"
corpus_file = "data/networks/lfr8-corpus.csv"
dict_file = "data/networks/lfr8-dict.csv"
header = False

def edges_to_corpus():
    edges = {}
    ids = set()
    print "Reading edges"
    with open(edge_file, "rb") as f_edges:
        if header:
            f_edges.next()
        for i, row in enumerate(f_edges):
            if i % 1000000 == 0:
                print "  row: %d" % i
            if row[0] == "#":
                continue
            source, target = row.rstrip().split(' ')
            source = int(source)
            target = int(target)
            ids.add(source)
            ids.add(target)
            try:
                source_edges = edges[source]
            except KeyError:
                source_edges = []
                edges[source] = source_edges
            source_edges.append(target)
            try:
                target_edges = edges[target]
            except KeyError:
                target_edges = []
                edges[target] = target_edges
            target_edges.append(source)
    print "Creating mapping"
    id_order = sorted(list(ids))
    min_id = min(id_order)
    max_id = max(id_order)
    print "  %d node_ids, %d - %d" % (len(id_order), min_id, max_id)
    print "Writing dict"
    id_to_index = {}
    with open(dict_file, "wb") as f_dict:
        f_dict.write("node_index,node_id\n")
        for i, node_id in enumerate(id_order):
            id_to_index[node_id] = i
            f_dict.write("%d,%d\n" % (i, node_id))
    print "Writing corpus"
    with open(corpus_file, "wb") as f_corpus:
        for i, node_id in enumerate(id_order):
            node_indexes = [str(id_to_index[v]) for v in edges[node_id]]
            node_indexes = [str(id_to_index[node_id])] + node_indexes
            document = "\t".join(node_indexes) + "\n"
            f_corpus.write(document)
            
class LFR8Corpus(object):
    def __iter__(self):
        for line in open(corpus_file, "rb"):
            nodes = [int(v) for v in line.rstrip().split("\t")]
            yield [(v, 1) for v in nodes]