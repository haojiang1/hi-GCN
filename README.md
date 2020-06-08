# hi-GCN
This is a Pytorch implementation of hierarchical Graph Convolutional Networks, as described in our paper.
# Requirement
tensorflow  
networkx
# Data
In order to use your own data, you have to provide  
an N by N adjacency matrix (N is the number of nodes),  
an N by D feature matrix (D is the number of features per node), and  
an N by E binary label matrix (E is the number of classes).  
