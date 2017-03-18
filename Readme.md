## Estimate SSN-LDA (1) Dirichlet parameters from node communities.
The estimation uses the MLE method described by (2) and implemented by (https://github.com/ericsuh/dirichlet).

The input csv should be comma separated with no spaces and a header row.
All ids should be numeric and start at 0.
The strength of a node's community membership (`member_prob`) should be a real number
between 0 and 1.
The format is:

    node_id,community_id,member_prob

If the csv containing the community data is `examples/example.csv`, then the usage is:

    python communityprior.py examples/example.csv alpha.csv beta.csv
    
Two files containing the priors, `alpha.csv` and `beta.csv`, will be written to the current directory.

### References
1. Zhang, H., Qiu, B., Giles, C. L., Foley, H. C., & Yen, J. (2007, May). An LDA-based community structure discovery approach for large-scale social networks. In _Intelligence and Security Informatics_, 2007 IEEE (pp. 200-207). IEEE.  
2. Minka, T. (2000). Estimating a Dirichlet distribution.
