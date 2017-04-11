import logging
import sys
import time
import gensim
import gensim.models
import corpus.WPCorpus

dict_file = "data/networks/wpuser-dict.csv"
out_file = "output/communities/wpusertalk-simplelda-%d-%s.csv"
# Ground truth: 12093
num_topics = int(sys.argv[1])
num_words = 6470

logging.basicConfig(filename='logs/gensim-wpusertalk-simple-%d.log' % num_topics, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
wpc = corpus.WPCorpus.WPCorpus()
wpm = gensim.models.LdaModel(wpc, num_topics=num_topics)

timestamp = time.strftime("%m%dT%H%M")

# Load dictionary
index_to_id = {}
with open(dict_file, "rb") as f_dict:
    f_dict.next()
    for row in f_dict:
        node_index, node_id = row.rstrip().split(",")
        index_to_id[int(node_index)] = int(node_id)

with open(out_file % (num_topics, timestamp), "wb") as f_out:
    f_out.write("node_id,community_id,member_prob\n")
    for topic in range(num_topics):
        try:
            weights = dict([(int(x[0]), x[1]) for x in wpm.show_topic(topic, num_words)])
        except IndexError:
            # Returned fewer topics than we asked for
            break
        print min(weights.keys()), " ", max(weights.keys())
        for i in range(len(weights)):
            node_id = index_to_id[i]
            node_weight = weights[i]
            f_out.write("%d,%d,%s\n" % (node_id, topic, repr(node_weight)))
    