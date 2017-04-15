import logging
import sys
import time
import gensim
import gensim.models
import corpus.FlickrCorpus

dict_file = "data/networks/flickr-dict.csv"
out_file = "output/communities/flickr-simplelda-%d-%s-%s.csv"
num_topics = int(sys.argv[1])
try:
    alpha = sys.argv[2]
    prior = alpha + "-"
    if alpha == 'sym':
        alpha = None
except IndexError:
    alpha = None
    prior = "sym-"
try:
    beta = sys.argv[3]
    prior += sys.argv[3]
except IndexError:
    beta = None
    prior += "sym"
num_words = 6470

logging.basicConfig(filename='logs/gensim-flickr-simple-%d-%s.log' % (num_topics, prior), format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
c = corpus.FlickrCorpus.FlickrCorpus()
m = gensim.models.LdaModel(c, num_topics=num_topics, alpha=alpha, eta=beta)

timestamp = time.strftime("%m%dT%H%M")

# Load dictionary
index_to_id = {}
with open(dict_file, "rb") as f_dict:
    f_dict.next()
    for row in f_dict:
        node_index, node_id = row.rstrip().split(",")
        index_to_id[int(node_index)] = int(node_id)

with open(out_file % (num_topics, prior, timestamp), "wb") as f_out:
    f_out.write("node_id,community_id,member_prob\n")
    for topic in range(num_topics):
        try:
            weights = dict([(int(x[0]), x[1]) for x in m.show_topic(topic, num_words)])
        except IndexError:
            # Returned fewer topics than we asked for
            break
        for i in range(len(weights)):
            node_id = index_to_id[i]
            node_weight = weights[i]
            f_out.write("%d,%d,%s\n" % (node_id, topic, repr(node_weight)))
    