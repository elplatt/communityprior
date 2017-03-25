import logging
import sys
import gensim
import gensim.models
import corpus.WPCorpus

dict_file = "data/networks/wpuser-dict.csv"
out_file = "output/communities/wpusertalk-simplelda-%d.csv"
# Ground truth: 12093
num_topics = int(sys.argv[1])
num_words = 6470

logging.basicConfig(filename='logs/gensim-wpusertalk-simple-%d.log' % num_topics, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
wpc = corpus.WPCorpus.WPCorpus()
wpm = gensim.models.LdaModel(wpc, num_topics=num_topics)

# Load dictionary
index_to_id = {}
with open(dict_file, "rb") as f_dict:
    f_dict.next
    for row in f_dict:
        node_index, node_id = row.rstrip().split(",")
        index_to_id[node_index] = node_id

with open(out_file % num_topics, "wb") as f_out:
    f_out.write("node_id,community_id,member_prob\n")
    for topic in range(num_topics):
        try:
            weights = dict(wpm.show_topic(topic, num_words))
        except IndexError:
            # Returned fewer topics than we asked for
            break
        for i in range(len(weights)):
            f_out.write("%d,%d,%f\n" % (index_to_id[i], topic, weights[i]))
    