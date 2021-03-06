import logging
import sys
import gensim
import gensim.models
import numpy as np
import pandas as pd
import corpus.FlickrCorpus

dict_file = "data/networks/flickr-dict.csv"
out_file = "output/communities/flickr-hybrid-%s.csv"
alpha_file = "output/priors/flickr-%s-alpha.csv"
beta_file = "output/priors/flickr-%s-beta.csv"
log_file = 'logs/gensim-flickr-hybrid-%s.log'
base_method = sys.argv[1]

# Load priors
alpha_df = pd.DataFrame.from_csv(alpha_file % base_method, index_col=None)
alpha = alpha_df['alpha_k']
beta_df = pd.DataFrame.from_csv(beta_file % base_method, index_col=None)
beta = beta_df['beta_v']
num_topics = len(alpha)
num_words = len(beta)

# Double communities for nonoverlapping base methods
if sys.argv[2] == "double":
    print "Extending alpha vector"
    alpha2 = np.ones(num_topics) * 50.0 / float(num_topics)
    alpha = alpha.append(pd.Series(alpha2))
    num_topics = num_topics * 2

logging.basicConfig(filename=log_file % base_method, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
c = corpus.FlickrCorpus.FlickrCorpus()
m = gensim.models.LdaModel(c, num_topics=num_topics, alpha=list(alpha), eta=list(beta))

# Load dictionary
print "Loading dictionary"
index_to_id = {}
with open(dict_file, "rb") as f_dict:
    f_dict.next()
    for row in f_dict:
        node_index, node_id = row.rstrip().split(",")
        index_to_id[int(node_index)] = int(node_id)

print "Writing output"
with open(out_file % base_method, "wb") as f_out:
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
    