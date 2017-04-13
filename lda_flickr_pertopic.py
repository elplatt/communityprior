import logging
import sys
import gensim
import gensim.models
import numpy as np
import pandas as pd
import corpus.FlickrCorpus

dict_file = "data/networks/flickr-dict.csv"
out_file = "output/communities/flickr-hybrid-%s-%s-pertopic.csv"
alpha_file = "output/priors/flickr-%s-alpha-pertopic.csv"
beta_file = "output/priors/flickr-%s-beta-pertopic.csv"
base_method = sys.argv[1]

# Load priors
priors = sys.argv[3]
alpha_df = pd.DataFrame.from_csv(alpha_file % base_method, index_col=None)
alpha = alpha_df['alpha_k']
beta = np.loadtxt(beta_file % base_method)
#beta_df = pd.DataFrame.from_csv(beta_file % base_method, index_col=None)
#beta = beta_df['beta_v']
num_topics = len(alpha)
num_words = beta.shape[1]

# Double communities for nonoverlapping base methods
if sys.argv[2] == "double":
    print "Extending alpha vector"
    alpha = 0.5 * alpha
    alpha2 = np.ones(num_topics) * 0.5 / float(num_topics)
    alpha = alpha.append(pd.Series(alpha2))
    print "Extending beta vector"
    beta = 0.5 * beta
    beta2 = np.ones(beta.shape) * 0.5 / float(num_topics)
    beta = np.concatenate((beta, beta2),axis=0)
    # Update num_topics
    num_topics = num_topics * 2


logging.basicConfig(filename='logs/gensim-flickr-hybrid-%s.log' % base_method, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
c = corpus.FlickrCorpus.FlickrCorpus()
if priors == "alphabeta":
    m = gensim.models.LdaModel(c, num_topics=num_topics, alpha=list(alpha), eta=list(beta))
elif priors == "alpha":
    m = gensim.models.LdaModel(c, num_topics=num_topics, alpha=list(alpha))
elif priors == "beta":
    m = gensim.models.LdaModel(c, num_topics=num_topics, beta=list(beta))

# Load dictionary
print "Loading dictionary"
index_to_id = {}
with open(dict_file, "rb") as f_dict:
    f_dict.next()
    for row in f_dict:
        node_index, node_id = row.rstrip().split(",")
        index_to_id[int(node_index)] = int(node_id)

print "Writing output"
with open(out_file % (base_method, priors), "wb") as f_out:
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
    