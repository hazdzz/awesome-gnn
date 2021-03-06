# GNNs_and_related_works_list

## About
A list for GNNs and related works.

## List for GNNs
### Spectral domain GNNs
| Number | GNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | Spectral CNN | Spectral Networks and Deep Locally Connected Networks on Graphs |  |
| 2 | ChebyNet | Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering | https://github.com/mdeff/cnn_graph |
| 3 | GCN | Semi-Supervised Classification with Graph Convolutional Networks | https://github.com/tkipf/pygcn |
| 4 | CayleyNet | CayleyNets: Graph Convolutional Neural Networks with Complex Rational Spectral Filters | https://github.com/amoliu/CayleyNet |
| 5 | LanczosNet | LanczosNet: Multi-Scale Deep Graph Convolutional Networks | https://github.com/lrjconan/LanczosNetwork |
| 6 | GWNN | Graph Wavelet Neural Network | https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork |

### Spatial/Vertex domain GNNs
| Number | GNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | DCNN | Diffusion-Convolutional Neural Networks | https://github.com/RicardoZiTseng/dcnn-tensorflow |
| 2 | GraphSAGE | Inductive Representation Learning on Large Graphs | https://github.com/williamleif/GraphSAGE |
| 3 | PinSAGE | Graph Convolutional Neural Networks for Web-Scale Recommender Systems | https://github.com/breadbread1984/PinSage-tf2.0 |
| 4 | MoNet | Geometric deep learning on graphs and manifolds using mixture model CNNs | https://github.com/theswgong/MoNet |
| 5 | GAT | Graph Attention Network | https://github.com/PetarV-/GAT |
| 6 | GaAN | GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs | https://github.com/jennyzhang0215/GaAN |
| 7 | FastGCN | FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling | https://github.com/matenure/FastGCN |
| 8 | GIN | How Powerful are Graph Neural Networks? | https://github.com/weihua916/powerful-gnns |
| 9 | SGC | Simplifying Graph Convolutional Networks | https://github.com/Tiiiger/SGC |
| 10 | gfNN | Revisiting Graph Neural Networks: All We Have is Low-Pass Filters | https://github.com/gear/gfnn |
| 11 | GraphHeat | Graph Convolutional Networks using Heat Kernel for Semi-supervised Learning | https://github.com/Eilene/GraphHeat |
| 12 | GDC | Diffusion Improves Graph Learning | https://github.com/klicperajo/gdc |
| 13 | PPNP & APPNP | Predict then Propagate: Graph Neural Networks meet Personalized PageRank | https://github.com/klicperajo/ppnp |
| 14 | DAGNN | Towards Deeper Graph Neural Networks | https://github.com/divelab/DeeperGNN |
| 15 | AFGNN | Demystifying Graph Neural Networks with Graph Filter Assessment | https://github.com/conferencesub/ICLR_2020 |
| 16 | GCNII | Simple and Deep Graph Convolutional Networks | https://github.com/chennnM/GCNII |
| 17 | SSGC | Simple Spectral Graph Convolution | https://github.com/allenhaozhu/SSGC |
| 18 | FAGCN | Beyond Low-frequency Information in Graph Convolutional Networks | https://github.com/bdy9527/FAGCN |
| 19 | GEOM-GCN | Geom-GCN: Geometric Graph Convolutional Networks | https://github.com/graphdml-uiuc-jlu/geom-gcn |
| 20 | GPR-GNN | Adaptive Universal Generalized PageRank Graph Neural Network | https://github.com/jianhao2016/GPRGNN |
| 21 | sRMGCNN | Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks | https://github.com/fmonti/mgcnn |

#### MPNNs
| Number | MPNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | MPNN | Neural Message Passing for Quantum Chemistry | https://github.com/brain-research/mpnn |
| 2 | SMP | Building powerful and equivariant graph neural networks with structural message-passing | https://github.com/cvignac/SMP |

## List for Over-smoothing
### Vanishing gradient problem
| Number | method | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 |  | Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning |
| 2 | DeepGCN | DeepGCNs: Can GCNs Go as Deep as CNNs? |  |
| 3 | | A Note on Over-Smoothing for Graph Neural Networks | |
| 4 | | Revisiting Over-smoothing in Deep GCNs | |
| 5 | DropEdge | DropEdge: Towards Deep Graph Convolutional Networks on Node Classification | https://github.com/DropEdge/DropEdge |
| 6 | DropEdge | Tackling Over-Smoothing for General Graph Convolutional Networks | https://github.com/DropEdge/DropEdge |
| 7 | PairNorm | PairNorm: Tackling Oversmoothing in GNNs | https://github.com/LingxiaoShawn/PairNorm |
| 8 | NodeNorm | Understanding and Resolving Performance Degradation in Graph Convolutional Networks | https://github.com/miafei/NodeNorm
| 9 | DGN | Towards Deeper Graph Neural Networks with Differentiable Group Normalization | https://github.com/Kaixiong-Zhou/DGN |

### Other
| Number | Method or GNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | BBGDC (beta-Bernoulli Graph DropConnect) | Bayesian Graph Neural Networks with Adaptive Connection Sampling | https://github.com/armanihm/GDC |
| 2 | LADIES | Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks | https://github.com/acbull/LADIES |
| 3 | | Graph Neural Networks Exponentially Lose Expressive Power for Node Classification | https://github.com/delta2323/gnn-asymptotics |
| 4 | JK-Net | Representation Learning on Graphs with Jumping Knowledge Networks |  |
| 5 | SSGC | Simple Spectral Graph Convolution | https://github.com/allenhaozhu/SSGC |
| 6 | GCNII | Simple and Deep Graph Convolutional Networks | https://github.com/chennnM/GCNII |
| 7 | GPR-GNN | Adaptive Universal Generalized PageRank Graph Neural Network | https://github.com/jianhao2016/GPRGNN |
| 8 | FAGCN | Beyond Low-frequency Information in Graph Convolutional Networks | https://github.com/bdy9527/FAGCN |

## Graph pooling layers
| Number | Graph pooling layer | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | DiffPool | Hierarchical Graph Representation Learning with Differentiable Pooling | https://github.com/RexYing/diffpool |
| 2 | g-U-Net | Graph U-Nets | https://github.com/HongyangGao/Graph-U-Nets |
| 3 | MinCutPool | Spectral Clustering with Graph Neural Networks for Graph Pooling | https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling |

## Others
| Number | GNN or method | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 |  | Graph Structure of Neural Networks |  |
| 2 | VGAE | Variational Graph Auto-Encoders | https://github.com/tkipf/gae |
| 3 | GC-MC | Graph Convolutional Matrix Completion |  |
| 4 | HGT | Heterogeneous Graph Transformer | https://github.com/acbull/pyHGT |
| 5 |  | Contrastive Multi-View Representation Learning on Graphs | https://github.com/kavehhassani/mvgrl |
| 6 | CS-GNN | Measuring and Improving the Use of Graph Information in Graph Neural Networks | https://github.com/yifan-h/CS-GNN |
| 7 | Benchmarking GNNs| Benchmarking Graph Neural Networks | https://github.com/graphdeeplearning/benchmarking-gnns |
| 8 | | Interpreting and Unifying Graph Neural Networks with An Optimization Framework | |

## Survey
| Number | survey | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 |  | Graph Neural Networks: A Review of Methods and Applications |  |
| 2 |  | A Comprehensive Survey on Graph Neural Networks |  |
