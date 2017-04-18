import logging
import sys
import gensim
import gensim.models
import numpy as np
import pandas as pd
import corpus.LFR0Corpus

dict_file = "data/networks/lfr0-dict.csv"
out_file = "output/communities/lfr0-hybrid-%s-%s-pertopic.csv"
alpha_file = "output/priors/lfr0-%s-alpha-pertopic.csv"
beta_file = "output/priors/lfr0-%s-beta-pertopic.csv"
log_file = 'logs/gensim-lfr0-hybrid-%s-%s.log'
base_method = sys.argv[1]
Corpus = corpus.LFR0Corpus.LFR0Corpus

# Load and configure priors
beta = np.loadtxt(beta_file % base_method)
num_topics, num_words = beta.shape
if sys.argv[3] == 'prior':
    alpha_df = pd.DataFrame.from_csv(alpha_file % base_method, index_col=None)
    alpha = list(alpha_df['alpha_k'])
elif sys.argv[3] == 'sym':
    alpha = None
else:
    alpha = sys.argv[3]
if sys.argv[4] == 'prior':
    pass
elif sys.argv[4] == 'sym':
    beta = None
else:
    beta = sys.argv[4]
priors = "%s-%s" % (sys.argv[3], sys.argv[4])

# Double communities for nonoverlapping base methods
if sys.argv[2] == "double":
    if not isinstance(alpha, basestring):
        print "Extending alpha vector"
        alpha = [0.5*a for a in alpha]
        alpha2 = np.ones(num_topics) * 0.5 / float(num_topics)
        alpha = list(alpha.append(pd.Series(alpha2)))
    if not isinstance(beta, basestring):
        print "Extending beta vector"
        beta = [0.5*b for b in beta]
        beta2 = np.ones(len(beta)) * 0.5 / float(num_topics)
        beta = list(np.concatenate((beta, beta2),axis=0))
    # Update num_topics
    num_topics = num_topics * 2

logging.basicConfig(filename=log_file % (base_method, priors), format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
c = Corpus()
m = gensim.models.LdaModel(c, num_topics=num_topics, alpha=alpha, eta=beta)

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
    
