edge_file = "data/networks/com-lj.ungraph.txt"
corpus_file = "data/networks/lj-corpus.csv"
dict_file = "data/networks/lj_dict.csv"

def edges_to_corpus():
    edges = {}
    print "Reading edges"
    with open(edge_file, "rb") as f_edges:
        for i, row in enumerate(f_edges):
            if i % 100000 == 0:
                print "  row: %d" % i
            if row[0] == "#":
                continue
            source, target = row.rstrip().split('\t')
            source = int(source)
            target = int(target)
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
    edge_keys = sorted(edges.keys())
    min_id = min(edge_keys)
    max_id = max(edge_keys)
    print "  %d node_ids, %d - %d" % (len(edge_keys), min_id, max_id)
    print "Writing dict"
    id_to_index = {}
    with open(dict_file, "wb") as f_dict:
        f_dict.write("node_index,node_id\n")
        for i, key in enumerate(edge_keys):
            id_to_index[key] = i
            f_dict.write("%d,%d\n" % (i, key))
    print "Writing corpus"
    with open(corpus_file, "wb") as f_corpus:
        for i, key in enumerate(edge_keys):
            node_indexes = [str(id_to_index[v]) for v in edges[key]]
            document = "\t".join(node_indexes) + "\n"
            f_corpus.write(document)
            
class LJCorpus(object):
    def __iter__(self):
        for line in open(corpus_file, "rb"):
            nodes = [int(v) for v in line.rstrip().split("\t")]
            yield [(v, 1) for v in nodes]