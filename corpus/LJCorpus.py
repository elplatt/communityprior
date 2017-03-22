edge_file = "data/networks/com-lj.ungraph.txt"
corpus_file = "data/networks/lj-corpus.csv"

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
    print "Writing corpus"
    with open(corpus_file, "wb") as f_corpus:
        for key in sorted(edges.keys()):
            document = "\t".join(edges[key]) + "\n"
            corpus_file.write(document)
            
class LJCorpus(object):
    def __iter__(self):
        for line in open(source, "rb"):
            if line[0] == "#":
                continue
