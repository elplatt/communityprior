import logging
import sys
import gensim
import gensim.models
import corpus.FlickrCorpus

dict_file = "data/networks/flickr-dict.csv"
out_file = "output/communities/flickr-simplelda-%d.csv"
num_topics = int(sys.argv[1])
num_words = 35275

logging.basicConfig(filename='logs/gensim-wpusertalk-simple-%d.log' % num_topics, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
c = corpus.FlickrCorpus.FlickrCorpus()
m = gensim.models.LdaModel(c, num_topics=num_topics)

# Load dictionary
index_to_id = {}
with open(dict_file, "rb") as f_dict:
    f_dict.next()
    for row in f_dict:
        node_index, node_id = row.rstrip().split(",")
        index_to_id[int(node_index)] = int(node_id)

with open(out_file % num_topics, "wb") as f_out:
    f_out.write("node_id,community_id,member_prob\n")
    for topic in range(num_topics):
        try:
            weights = dict([(int(x[0]), x[1]) for x in m.show_topic(topic, num_words)])
        except IndexError:
            # Returned fewer topics than we asked for
            break
        print min(weights.keys()), " ", max(weights.keys())
        for i in range(len(weights)):
            node_id = index_to_id[i]
            node_weight = weights[i]
            f_out.write("%d,%d,%f\n" % (node_id, topic, node_weight))
    