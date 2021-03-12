#Model_1 is purely homemade GCN
#Model_2 is investigation of TGCN, shows not too bad results, so:
#Model_3 is a 'better' implementation
#Model_4 is playing with another scatter_distribution function that returns skew and kurtosis instead of max, min
#Model_5 FeaStConv
#Model_7 Tried different stuff, firstly added a central node, with edges to/from all other nodes on which message passing is done through a GRUcell, this works nicely. I then tried removing the edge_attr_update and later the node_update, which both did not affect the accuracy.
#(this is reflected in m71 on wandb)
#(m72 is the same model but without any dependancy on pre-defined edge_indeces)
#(m73 is then the same model but with feature_construction included)
#(m732 is switching the concatenation when doing message pass the other way)
#Model_9 is Model_7 but 'improved' with batchnormalization and perhaps skiplayer and so forth