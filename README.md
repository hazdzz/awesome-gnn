# GNNs and related works list

## About
A list for GNNs and related works.

## List for GNNs
### Spectral domain GNNs
#### Convolutional GNNs
| Number | GNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | Spectral CNN | Spectral Networks and Locally Connected Networks on Graphs | |
| 2 | ChebyNet | Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering | https://github.com/mdeff/cnn_graph |
| 3 | GCN | Semi-Supervised Classification with Graph Convolutional Networks | https://github.com/tkipf/gcn |
| 4 | CayleyNet | CayleyNets: Graph Convolutional Neural Networks with Complex Rational Spectral Filters | https://github.com/amoliu/CayleyNet |
| 5 | LanczosNet | LanczosNet: Multi-Scale Deep Graph Convolutional Networks | https://github.com/lrjconan/LanczosNetwork |
| 6 | Bayesian GCN | Bayesian graph convolutional neural networks for semi-supervised classification | https://github.com/huawei-noah/BGCN |
| 7 | GWNN | Graph Wavelet Neural Network | https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork |
| 8 | GraphWave | Learning Structural Node Embeddings via Diffusion Wavelets | https://github.com/benedekrozemberczki/GraphWaveMachine |
| 9 | GraphHeat | Graph Convolutional Networks using Heat Kernel for Semi-supervised Learning | https://github.com/Eilene/GraphHeat |
| 10 | HANet | Fast Haar Transforms for Graph Neural Networks | |
| 11 | MathNet | MathNet: Haar-Like Wavelet Multiresolution-Analysis for Graph Representation and Learning | |
| 12 | | Graph Neural Networks with convolutional ARMA filters | |
| 13 | DGCN | Spectral-based Graph Convolutional Network for Directed Graphs | |
| 14 | DGCN | Directed Graph Convolutional Network | |
| 15 | MotifNet | MotifNet: a motif-based Graph Convolutional Network for directed graphs | |
| 16 | FisherGCN | Fisher-Bures Adversary Graph Convolutional Networks | https://github.com/D61-IA/FisherGCN |
| 17 | DFNet | DFNets: Spectral CNNs for Graphs with Feedback-Looped Filters | https://github.com/wokas36/DFNets |
| 18 | MagNet | MagNet: A Neural Network for Directed Graphs | https://github.com/matthew-hirn/magnet |

#### Attentional GNNs
| Number | GNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | SpGAT | Spectral Graph Attention Network | |

#### Graph Pooling (Graph Coarsening)
| Number | Graph Pooling | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | LaPool | Towards Interpretable Sparse Graph Representation Learning with Laplacian Pooling | |
| 2 | EigenPooling | Graph Convolutional Networks with EigenPooling | https://github.com/alge24/eigenpooling |
| 3 | HaarPool | Haar Graph Pooling | https://github.com/YuGuangWang/HaarPool |

### Spatial/Vertex domain GNNs
#### Convolutional GNNs
| Number | GNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | DCNN | Diffusion-Convolutional Neural Networks | https://github.com/RicardoZiTseng/dcnn-tensorflow |
| 2 | MoNet | Geometric deep learning on graphs and manifolds using mixture model CNNs | https://github.com/theswgong/MoNet |
| 3 | FastGCN | FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling | https://github.com/matenure/FastGCN |
| 4 | GraphSAGE | Inductive Representation Learning on Large Graphs | https://github.com/williamleif/GraphSAGE |
| 5 | PinSAGE | Graph Convolutional Neural Networks for Web-Scale Recommender Systems | |
| 6 | GIN | How Powerful are Graph Neural Networks? | https://github.com/weihua916/powerful-gnns |
| 7 | SGC | Simplifying Graph Convolutional Networks | https://github.com/Tiiiger/SGC |
| 8 | gfNN | Revisiting Graph Neural Networks: All We Have is Low-Pass Filters | https://github.com/gear/gfnn |
| 9 | JK-Net | Representation Learning on Graphs with Jumping Knowledge Networks | |
| 10 | Geom-GCN | Geom-GCN: Geometric Graph Convolutional Networks | https://github.com/graphdml-uiuc-jlu/geom-gcn |
| 11 | sRMGCNN | Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks | https://github.com/fmonti/mgcnn |
| 12 | GDC | Diffusion Improves Graph Learning | https://github.com/klicperajo/gdc |
| 13 | PPNP & APPNP | Predict then Propagate: Graph Neural Networks meet Personalized PageRank | https://github.com/klicperajo/ppnp |
| 14 | DAGNN | Towards Deeper Graph Neural Networks | https://github.com/divelab/DeeperGNN |
| 15 | AFGNN | Demystifying Graph Neural Networks with Graph Filter Assessment | https://github.com/conferencesub/ICLR_2020 |
| 16 | GCNII | Simple and Deep Graph Convolutional Networks | https://github.com/chennnM/GCNII |
| 17 | SSGC | Simple Spectral Graph Convolution | https://github.com/allenhaozhu/SSGC |
| 18 | GPR-GNN | Adaptive Universal Generalized PageRank Graph Neural Network | https://github.com/jianhao2016/GPRGNN |
| 19 | | Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks | https://github.com/chrsmrrs/k-gnn |
| 20 | Cluster-GCN | Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks | https://github.com/google-research/google-research/tree/master/cluster_gcn |
| 21 | SIGN | SIGN: Scalable Inception Graph Neural Networks | https://github.com/twitter-research/sign |
| 22 | DGN | Directional Graph Networks | https://github.com/Saro00/DGN |

#### Attentional GNNs
| Number | GNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | AGNN | Attention-based Graph Neural Network for Semi-supervised Learning | https://github.com/dawnranger/pytorch-AGNN |
| 2 | GAT | Graph Attention Network | https://github.com/PetarV-/GAT |
| 3 | MCN | Higher-order Graph Convolutional Networks | |
| 4 | CS-GNN | Measuring and Improving the Use of Graph Information in Graph Neural Networks | https://github.com/yifan-h/CS-GNN |
| 5 | MixHop | MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing | https://github.com/samihaija/mixhop |
| 6 | GaAN | GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs | https://github.com/jennyzhang0215/GaAN |
| 7 | hGANet | Graph Representation Learning via Hard and Channel-Wise Attention Networks | |
| 8 | GAM | Graph Classification using Structural Attention | https://github.com/benedekrozemberczki/GAM |
| 9 | C-GAT | Improving Graph Attention Networks with Large Margin-based Constraints | |
| 10 | FAGCN | Beyond Low-frequency Information in Graph Convolutional Networks | https://github.com/bdy9527/FAGCN |
| 11 | RGCN and RGAT | Relational Graph Attention Networks | https://github.com/babylonhealth/rgat |
| 12 | DPGAT and GATv2 | How Attentive are Graph Attention Networks? | https://github.com/tech-srl/how_attentive_are_gats |

#### Graph Pooling (Graph Coarsening)
| Number | Graph Pooling | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | SortPooling | An End-to-End Deep Learning Architecture for Graph Classification | https://github.com/muhanzhang/DGCNN |
| 2 | DiffPool | Hierarchical Graph Representation Learning with Differentiable Pooling | https://github.com/RexYing/diffpool |
| 3 | gPool and gUnpool | Graph U-Nets | https://github.com/HongyangGao/Graph-U-Nets |
| 4 | SAGPool | Self-Attention Graph Pooling | https://github.com/inyeoplee77/SAGPool |
| 5 | Relational Pooling | Relational Pooling for Graph Representations | https://github.com/PurdueMINDS/RelationalPooling |
| 6 | HPL-SL | Hierarchical Graph Pooling with Structure Learning | https://github.com/cszhangzhen/HGP-SL |
| 7 | StructPool | StructPool: Structured Graph Pooling via Conditional Random Fields | https://github.com/Nate1874/StructPool |
| 8 | MinCutPool | Spectral Clustering with Graph Neural Networks for Graph Pooling | https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling |
| 9 | GMT | Accurate Learning of Graph Representations with Graph Multiset Pooling | https://github.com/JinheonBaek/GMT |
| 10 | GSAPool | Structure-Feature based Graph Self-adaptive Pooling | |
| 11 | NDP | Hierarchical Representation Learning in Graph Neural Networks with Node Decimation Pooling | https://github.com/danielegrattarola/decimation-pooling |
| 12 | VIPool | Graph Cross Networks with Vertex Infomax Pooling | https://github.com/limaosen0/GXN |
| 13 | iPool | iPool—Information-Based Pooling in Hierarchical Graph Neural Networks | |

#### MPNNs
| Number | MPNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | MPNN | Neural Message Passing for Quantum Chemistry | https://github.com/brain-research/mpnn |
| 2 | SMP | Building powerful and equivariant graph neural networks with structural message-passing | https://github.com/cvignac/SMP |
| 3 | PNA | Principal Neighbourhood Aggregation for Graph Nets | https://github.com/lukecavabarrett/pna |
| 4 | EGC-S and EGC-M | Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions | https://github.com/shyam196/egc |

#### Heterogeneous Graph Neural Networks
| Number | HGNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | HetGNN | Heterogeneous Graph Neural Network | https://github.com/chuxuzhang/KDD2019_HetGNN |
| 2 | HAN | Heterogeneous Graph Attention Network | https://github.com/Jhy1993/HAN |
| 3 | HGT | Heterogeneous Graph Transformer | https://github.com/acbull/pyHGT |
| 4 | MAGNN | MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding | https://github.com/cynricfu/MAGNN |
| 5 | HeCo | Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning | https://github.com/liun-online/HeCo |

#### Hyperbolic Graph Neural Networks
| Number | HGNN or method | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | HGCN | Hyperbolic Graph Convolutional Neural Networks | https://github.com/HazyResearch/hgcn |
| 2 | HGNN | Hyperbolic Graph Neural Networks | https://github.com/facebookresearch/hgnn |
| 3 | HAT | Hyperbolic Graph Attention Network | |
| 4 | HNN + HBN | Differentiating through the Fréchet Mean | https://github.com/CUAI/Differentiable-Frechet-Mean |
| 5 | LGCN | Lorentzian Graph Convolutional Networks | |
| 6 | H2H-GCN | A Hyperbolic-to-Hyperbolic Graph Convolutional Network | |
| 7 | HGCF | HGCF: Hyperbolic Graph Convolution Networks for Collaborative Filtering | https://github.com/layer6ai-labs/HGCF |

#### Capsule Graph Neural Network
| Number | CGNN or method | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | CapsGNN | Capsule Graph Neural Network | https://github.com/benedekrozemberczki/CapsGNN |
| 2 | GCAPS-CNN | Graph Capsule Convolutional Neural Networks | https://github.com/vermaMachineLearning/Graph-Capsule-CNN-Networks/ |

#### Graph Neural ODE or PDE
| Number | GNODE or GNPDE | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | | Graph Neural Ordinary Differential Equations | https://github.com/Zymrael/gde |
| 2 | | Ordinary differential equations on graph networks | |
| 3 | CGF | Continuous Graph Flow | |
| 4 | CGNN | Continuous Graph Neural Networks | https://github.com/DeepGraphLearning/ContinuousGNN |
| 5 | NDCN | Neural Dynamics on Complex Networks | https://github.com/calvin-zcx/ndcn |
| 6 | CFD-GCN | Combining Differentiable PDE Solvers and Graph Neural Networks for Fluid Flow Prediction | https://github.com/locuslab/cfd-gcn |
| 7 | GRAND | GRAND: Graph Neural Diffusion | https://github.com/twitter-research/graph-neural-pde |
| 8 | DeltaGN and OGN | Hamiltonian Graph Networks with ODE Integrators | |

## List for Over-smoothing
### Analyses
| Number | Paper | Code |
|:------:|--------------------------|-------|
| 1 | Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning | |
| 2 | DeepGCNs: Can GCNs Go as Deep as CNNs? | |
| 3 | Graph Neural Networks Exponentially Lose Expressive Power for Node Classification | https://github.com/delta2323/gnn-asymptotics |
| 4 | A Note on Over-Smoothing for Graph Neural Networks | |
| 5 | Revisiting Over-smoothing in Deep GCNs | |
| 6 | Measuring and Improving the Use of Graph Information in Graph Neural Networks | https://github.com/yifan-h/CS-GNN |
| 7 | Simple and Deep Graph Convolutional Networks | https://github.com/chennnM/GCNII |

### Dropout-like
| Number | Method | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | DropEdge | DropEdge: Towards Deep Graph Convolutional Networks on Node Classification | https://github.com/DropEdge/DropEdge |
| 2 | DropEdge | Tackling Over-Smoothing for General Graph Convolutional Networks | https://github.com/DropEdge/DropEdge |

### Graph Normalization
| Number | Norm | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | PairNorm | PairNorm: Tackling Oversmoothing in GNNs | https://github.com/LingxiaoShawn/PairNorm |
| 2 | NodeNorm | Understanding and Resolving Performance Degradation in Graph Convolutional Networks | https://github.com/miafei/NodeNorm |
| 3 | DGN | Towards Deeper Graph Neural Networks with Differentiable Group Normalization | https://github.com/Kaixiong-Zhou/DGN |
| 4 | GraphNorm | GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training | https://github.com/lsj2408/GraphNorm |

### Sampling
| Number | Method or GNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | FastGCN | FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling | https://github.com/matenure/FastGCN |
| 2 | VR-GCN | Stochastic Training of Graph Convolutional Networks with Variance Reduction | https://github.com/thu-ml/stochastic_gcn |
| 2 | | Advancing GraphSAGE with A Data-driven Node Sampling | https://github.com/oj9040/GraphSAGE_RL |
| 3 | | Adaptive Sampling Towards Fast Graph Representation Learning | https://github.com/huangwb/AS-GCN |
| 4 | BBGDC (beta-Bernoulli Graph DropConnect) | Bayesian Graph Neural Networks with Adaptive Connection Sampling | https://github.com/armanihm/GDC |
| 5 | LADIES | Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks | https://github.com/acbull/LADIES |
| 6 | GraphSAINT | GraphSAINT: Graph Sampling Based Inductive Learning Method | https://github.com/GraphSAINT/GraphSAINT |
| 7 | MVS-GNN | Minimal Variance Sampling with Provable Guarantees for Fast Training of Graph Neural Networks | |

### Scattering
| Number | Method or GNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | Scattering GCN | Scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks | https://github.com/dms-net/scatteringGCN |

## List for Over-squashing
| Number | Method or GNN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | | On the Bottleneck of Graph Neural Networks and its Practical Implications | https://github.com/tech-srl/bottleneck/ |

## List for Graph Autoencoders (GAE)
| Number | GAE | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | DNGR | Deep Neural Networks for Learning Graph Representations | |
| 2 | SDNE | Structural Deep Network Embedding | |
| 3 | DVNE | Deep Variational Network Embedding in Wasserstein Space | |
| 4 | VGAE | Variational Graph Auto-Encoders | https://github.com/tkipf/gae | |
| 5 | GC-MC | Graph Convolutional Matrix Completion | https://github.com/riannevdberg/gc-mc |
| 6 | ARVGA | Adversarially regularized graph autoencoder for graph embedding | |
| 7 | NetRA | Learning deep network representations with adversarially regularized autoencoders | |
| 8 | DeepGMG | Learning deep generative models of graphs | |
| 9 | GraphRNN | GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models | |
| 10 | GraphVAE | Graphvae: Towards generation of small graphs using variational autoencoders | |
| 11 | | Constrained generation of semantically valid graphs via regularizing variational autoencoders | |
| 12 | Gravity Graph VAE and Gravity Graph AE | Gravity-Inspired Graph Autoencoders for Directed Link Prediction | https://github.com/deezer/gravity_graph_autoencoders |

## List for Graph Generative Adversarial Networks (GGAN)
| Number | GGAN | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | GraphGAN |GraphGAN: Graph Representation Learning with Generative Adversarial Nets |
| 2 | MolGAN | MolGAN: An implicit generative model for small molecular graphs |
| 3 | NetGAN | Netgan: Generating graphs via random walks |

## List for Graph MLP and Graph Transformer
| Number |  Graph MLP or Graph Transformer | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | Graph-MLP | Graph-MLP: Node Classification without Message Passing in Graph | https://github.com/yanghu819/Graph-MLP |
| 2 | Graphormer | Do Transformers Really Perform Bad for Graph Representation? | https://github.com/Microsoft/Graphormer |
| 3 | GraphiT | GraphiT: Encoding Graph Structure in Transformers | https://github.com/inria-thoth/GraphiT |

## List for graph pre-training
| Number | Pre-training mathod | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | | Strategies for Pre-training Graph Neural Networks | https://github.com/snap-stanford/pretrain-gnns/ |
| 2 | GCC | GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training | https://github.com/THUDM/GCC |

## List for GNN explainers
| Number | GNN or method | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | GNNExplainer | GNNExplainer: Generating Explanations for Graph Neural Networks | https://github.com/RexYing/gnn-model-explainer |
| 2 | PGExplainer | Parameterized Explainer for Graph Neural Network | https://github.com/flyingdoog/PGExplainer |
| 3 | PGM-Explainer | PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks | https://github.com/vunhatminh/PGMExplainer |
| 4 | XGNN | XGNN: Towards Model-Level Explanations of Graph Neural Networks | |
| 5 | Gem | Generative Causal Explanations for Graph Neural Networks | https://github.com/wanyu-lin/ICML2021-Gem | 

## List for Others
| Number | GNN or method | Paper | Code |
|:------:|:--------------------------:|-------|------|
| 1 | | Graph Structure of Neural Networks | |
| 2 | | Contrastive Multi-View Representation Learning on Graphs | https://github.com/kavehhassani/mvgrl |
| 3 | Benchmarking GNNs| Benchmarking Graph Neural Networks | https://github.com/graphdeeplearning/benchmarking-gnns |
| 4 | | Interpreting and Unifying Graph Neural Networks with An Optimization Framework | |
| 5 | | What graph neural networks cannot learn: depth vs width |
| 6 | | Extract the Knowledge of Graph Neural Networks and Go Beyond it: An Effective Knowledge Distillation Framework | https://github.com/BUPT-GAMMA/CPF |
| 7 | SUGAR | SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism | https://github.com/RingBDStack/SUGAR |
| 8 | | Towards Sparse Hierarchical Graph Classifiers | |
| 9 | OGB | Open Graph Benchmark: Datasets for Machine Learning on Graphs | https://github.com/snap-stanford/ogb |
| 10 | AdaGCN | AdaGCN: Adaboosting Graph Convolutional Networks into Deep Models | https://github.com/datake/AdaGCN |
| 11 | BGNN(BGCN, BGAT) | Bilinear Graph Neural Network with Neighbor Interactions | https://github.com/zhuhm1996/bgnn |
| 12 | RevGNN-Deep and RevGNN-Wide | Training Graph Neural Networks with 1000 Layers | https://github.com/lightaime/deep_gcns_torch/tree/master/examples/ogb_eff/ogbn_proteins |
| 13 | OGB-LSC | OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs | |
| 14 | DrGCNs | Dimensional Reweighting Graph Convolutional Networks | |
| 15 | GAS | GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings | https://github.com/rusty1s/pyg_autoscale |
| 16 | TWIRLS | Graph Neural Networks Inspired by Classical Iterative Algorithms | https://github.com/FFTYYY/TWIRLS |
| 17 | GAT-Lip | Lipschitz Normalization for Self-Attention Layers with Application to Graph Neural Networks | |
| 18 | | Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective | https://github.com/balcilar/gnn-spectral-expressive-power |
| 19 | | Deep Graph Neural Networks with Shallow Subgraph Samplers | https://github.com/facebookresearch/shaDow_GNN |
| 20 | | Large-scale graph representation learning with very deep GNNs and self-supervision | https://github.com/deepmind/deepmind-research/tree/master/ogb_lsc |
| 21 | GCN-LPA | Unifying Graph Convolutional Neural Networks and Label Propagation | https://github.com/hwwang55/GCN-LPA |
| 22 | L-GCN and L<sup>2</sup>-GCN | L<sup>2</sup>-GCN: Layer-Wise and Learned Efficient Training of Graph Convolutional Networks | https://github.com/VITA-Group/L2-GCN |
| 23 | | A Fair Comparison of Graph Neural Networks for Graph Classification | https://github.com/diningphil/gnn-comparison |

## List for Surveys
| Number | Paper |
|:------:|--------------------------|
| 1 | Graph Neural Networks: A Review of Methods and Applications |
| 2 | A Comprehensive Survey on Graph Neural Networks |
| 3 | Deep Learning on Graphs: A Survey |
| 4 | Graph Neural Networks in Recommender Systems: A Survey |
