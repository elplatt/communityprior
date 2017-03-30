import logging
import sys
import gensim
import gensim.models
import numpy as np
import pandas as pd
import corpus.WPCorpus

dict_file = "data/networks/wpuser-dict.csv"
out_file = "output/communities/wpusertalk-hybrid-%s.csv"
alpha_file = "data/priors/wikipedia-%s-alpha.csv"
beta_file = "data/priors/wikipedia-%s-beta.csv"
base_method = sys.argv[1]

# Load priors
alpha_df = pd.DataFrame.from_csv(alpha_file % base_method, index_col=None)
alpha = alpha_df['alpha_k']
beta_df = pd.DataFrame.from_csv(beta_file % base_method, index_col=None)
beta = beta_df['beta_v']
num_topics = len(alpha)
num_nodes = len(beta)

# Double communities for nonoverlapping base methods
if sys.argv[2] == "double":
    alpha2 = np.ones(num_topics) * 50.0 / float(num_coms)
    alpha = alpha.append(np.Series(alpha2))
    num_topics = num_topics * 2

logging.basicConfig(filename='logs/gensim-wpusertalk-hybrid-%s.log' % num_topics, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
c = corpus.WPCorpus.WPCorpus()
m = gensim.models.LdaModel(c, num_topics=num_topics, alpha=alpha, eta=beta)

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
            weights = dict([(int(x[0]), x[1]) for x in wpm.show_topic(topic, num_words)])
        except IndexError:
            # Returned fewer topics than we asked for
            break
        print min(weights.keys()), " ", max(weights.keys())
        for i in range(len(weights)):
            node_id = index_to_id[i]
            node_weight = weights[i]
            f_out.write("%d,%d,%f\n" % (node_id, topic, node_weight))
    