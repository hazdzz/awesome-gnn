# GNNs and related works list
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![forks](https://img.shields.io/github/forks/hazdzz/awesome-gnn)](https://github.com/hazdzz/awesome-gnn/network/members)
[![stars](https://img.shields.io/github/stars/hazdzz/awesome-gnn)](https://github.com/hazdzz/awesome-gnn/stargazers)
[![License](https://img.shields.io/github/license/hazdzz/awesome-gnn)](./LICENSE)

## About
A list for GNNs and related works.

## List for GNNs
### Spectral domain GNNs
#### Convolutional GNNs
##### Based on graph Fourier transform
| Number | GNN | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | Spectral CNN | Spectral Networks and Locally Connected Networks on Graphs | | ICLR 2014 | https://openreview.net/forum?id=DQNsQf-UsoDBa |
| 2 | | Deep Convolutional Networks on Graph-Structured Data | | | https://arxiv.org/abs/1506.05163 |
| 3 | ChebNet | Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering | https://github.com/mdeff/cnn_graph | NeurIPS 2016 | https://proceedings.neurips.cc/paper/2016/hash/04df4d434d481c5bb723be1b6df1ee65-Abstract.html |
| 4 | GCN | Semi-Supervised Classification with Graph Convolutional Networks | https://github.com/tkipf/gcn | ICLR 2017 | https://openreview.net/forum?id=SJU4ayYgl |
| 5 | SGC | Simplifying Graph Convolutional Networks | https://github.com/Tiiiger/SGC | ICML 2019 | http://proceedings.mlr.press/v97/wu19e.html |
| 6 | gfNN | Revisiting Graph Neural Networks: Graph Filtering Perspective | https://github.com/gear/gfnn | International Conference on Pattern Recognition 2021 | https://ieeexplore.ieee.org/document/9412278 |
| 7 | CayleyNet | CayleyNets: Graph Convolutional Neural Networks with Complex Rational Spectral Filters | | IEEE Transactions on Signal Processing | https://github.com/amoliu/CayleyNet | https://ieeexplore.ieee.org/abstract/document/8521593 |
| 8 | MotifNet | MotifNet: a motif-based Graph Convolutional Network for directed graphs | | 2018 IEEE Data Science Workshop | https://ieeexplore.ieee.org/abstract/document/8439897 |
| 9 | LanczosNet | LanczosNet: Multi-Scale Deep Graph Convolutional Networks | https://github.com/lrjconan/LanczosNetwork | ICLR 2019 | https://openreview.net/forum?id=BkedznAqKQ |
| 10 | PPNP & APPNP | Predict then Propagate: Graph Neural Networks meet Personalized PageRank | https://github.com/klicperajo/ppnp | ICLR 2019 | https://openreview.net/forum?id=H1gL-2A9Ym |
| 11 | GDC | Diffusion Improves Graph Learning | https://github.com/klicperajo/gdc | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html |
| 12 | GCNII | Simple and Deep Graph Convolutional Networks | https://github.com/chennnM/GCNII | ICML 2020 | https://proceedings.mlr.press/v119/chen20v.html |
| 13 | ARMA | Graph Neural Networks with convolutional ARMA filters |  https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional/arma_conv.py#L10 | IEEE Transactions on Pattern Analysis and Machine Intelligence | https://ieeexplore.ieee.org/abstract/document/9336270 |
| 14 | DFNet | DFNets: Spectral CNNs for Graphs with Feedback-Looped Filters | https://github.com/wokas36/DFNets | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/f87522788a2be2d171666752f97ddebb-Abstract.html |
| 15 | Snowball and Truncated Krylov | Break the Ceiling: Stronger Multi-scale Deep Graph Convolutional Networks | https://github.com/PwnerHarry/Stronger_GCN | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/ccdf3864e2fa9089f9eca4fc7a48ea0a-Abstract.html |
| 16 | BGNN | Bilinear Graph Neural Network with Neighbor Interactions | https://github.com/zhuhm1996/bgnn | IJCAI 2020 | https://www.ijcai.org/proceedings/2020/202 |
| 17 | GBP | Scalable Graph Neural Networks via Bidirectional Propagation | https://github.com/chennnM/GBP | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/a7789ef88d599b8df86bbee632b2994d-Abstract.html |
| 18 | FisherGCN | Fisher-Bures Adversary Graph Convolutional Networks | https://github.com/D61-IA/FisherGCN | ICML 2020 | http://proceedings.mlr.press/v115/sun20a.html |
| 19 | DGCN | Directed Graph Convolutional Network | | | https://arxiv.org/abs/2004.13970 |
| 20 | SIGN | SIGN: Scalable Inception Graph Neural Networks | | ICML 2020 Workshop | https://arxiv.org/abs/2004.11198 |
| 21 | DiGCN | Digraph Inception Convolutional Networks | https://github.com/flyingtango/DiGCN | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/cffb6e2288a630c2a787a64ccc67097c-Abstract.html |
| 22 | HKGCN | Generalizing Graph Convolutional Networks via Heat Kernel | | | https://openreview.net/forum?id=yBJihVXahXc |
| 23 | S<sup>2</sup>GC | Simple Spectral Graph Convolution | https://github.com/allenhaozhu/SSGC | ICLR 2021 | https://openreview.net/forum?id=CYO5T-YjWZV |
| 24 | GPR-GNN | Adaptive Universal Generalized PageRank Graph Neural Network | https://github.com/jianhao2016/GPRGNN | ICLR 2021 | https://openreview.net/forum?id=n6jl7fLxrP |
| 25 | GA-MLP | On Graph Neural Networks versus Graph-Augmented MLPs | https://github.com/leichen2018/GNN_vs_GAMLP | ICLR 2021 | https://openreview.net/forum?id=tiqI7w64JG2 |
| 26 | DGCN | On Provable Benefits of Depth in Training Graph Convolutional Networks | https://github.com/CongWeilin/DGCN | NeurIPS 2021 | https://openreview.net/forum?id=r-oRRT-ElX |
| 27 | MagNet | MagNet: A Neural Network for Directed Graphs | https://github.com/matthew-hirn/magnet | NeurIPS 2021 | https://openreview.net/forum?id=TRDAFiwDq8A |
| 28 | BernNet | BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation | https://github.com/ivam-he/BernNet | NeurIPS 2021 | https://openreview.net/forum?id=WigDnV-_Gq |
| 29 | EdgeNet | EdgeNets: Edge Varying Graph Neural Networks | | IEEE Transactions on Pattern Analysis and Machine Intelligence | https://ieeexplore.ieee.org/abstract/document/9536420 |
| 30 | AdaGNN | AdaGNN: Graph Neural Networks with Adaptive Frequency Response | https://github.com/yushundong/AdaGNN | CIKM 2021 | https://dl.acm.org/doi/abs/10.1145/3459637.3482226 |
| 31 | ADA-UGNN | A Unified View on Graph Neural Networks as Graph Signal Denoising | https://github.com/alge24/ADA-UGNN | CIKM 2021 | https://dl.acm.org/doi/abs/10.1145/3459637.3482225 |
| 32 | AirGNN | Graph Neural Networks with Adaptive Residual | https://github.com/lxiaorui/AirGNN | NeurIPS 2021 | https://proceedings.neurips.cc/paper/2021/hash/50abc3e730e36b387ca8e02c26dc0a22-Abstract.html |
| 33 | ADC | Adaptive Diffusion in Graph Neural Networks | https://github.com/abcbdf/ADC | NeurIPS 2021 | https://proceedings.neurips.cc/paper/2021/hash/c42af2fa7356818e0389593714f59b52-Abstract.html |
| 34 | U-GCN | Universal Graph Convolutional Networks | https://github.com/jindi-tju/U-GCN | NeurIPS 2021 | https://papers.nips.cc/paper/2021/hash/5857d68cd9280bc98d079fa912fd6740-Abstract.html |
| 35 | BM-GCN | Block Modeling-Guided Graph Convolutional Neural Networks | https://github.com/hedongxiao-tju/BM-GCN | AAAI 2022 | https://ojs.aaai.org/index.php/AAAI/article/view/20319 |
| 36 | Ortho-GConv | Orthogonal Graph Neural Networks | https://github.com/KaiGuo20/Ortho-GConv | AAAI 2022 | https://ojs.aaai.org/index.php/AAAI/article/view/20316 |
| 37 | <sup>p</sup>GNN | p-Laplacian Based Graph Neural Networks | https://github.com/guoji-fu/pgnns | ICML 2022 | https://proceedings.mlr.press/v162/fu22e.html |
| 38 | Spec-GN and Norm-GN | A New Perspective on the Effects of Spectrum in Graph Neural Networks | https://github.com/qslim/gnn-spectrum | ICML 2022 | https://proceedings.mlr.press/v162/yang22n.html | 
| 39 | JacobiConv | How Powerful are Spectral Graph Neural Networks | https://github.com/GraphPKU/JacobiConv | ICML 2022 | https://proceedings.mlr.press/v162/wang22am.html |
| 40 | G<sup>2</sup>CN | G<sup>2</sup>CN: Graph Gaussian Convolution Networks with Concentrated Graph Filters | | ICML 2022| https://proceedings.mlr.press/v162/li22h.html |
| 41 | ChebNetII | Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited | https://github.com/ivam-he/ChebNetII | NeurIPS 2022 | https://arxiv.org/abs/2202.03580 |
| 42 | ACMII-GCN | Revisiting Heterophily For Graph Neural Networks | https://github.com/SitaoLuan/ACM-GNN | NeurIPS 2022 | https://proceedings.neurips.cc/paper_files/paper/2022/hash/092359ce5cf60a80e882378944bf1be4-Abstract-Conference.html |
| 43 | FavardGNN and OptBasisGNN | Graph Neural Networks with Learnable and Optimal Polynomial Bases | https://github.com/yuziGuo/FarOptBasis | ICML 2023 | https://proceedings.mlr.press/v202/guo23i |
| 44 | Spectral-SGCN-I, Spectral-SGCN-II, Spectral-S2GCN, and Singned-Magnet | Signed Graph Neural Networks: A Frequency Perspective | https://github.com/rahulsinghchandraul/Spectral_Signed_GNN | TMLR 2023 | https://openreview.net/forum?id=RZveYHgZbu |

##### Based on graph Wavelet transform
| Number | GNN | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | HANet | Fast Haar Transforms for Graph Neural Networks | | Neural Networks | https://www.sciencedirect.com/science/article/abs/pii/S0893608020301568 |
| 2 | MathNet | MathNet: Haar-Like Wavelet Multiresolution-Analysis for Graph Representation and Learning | | | https://arxiv.org/abs/2007.11202 |
| 3 | UFGConv and UFGPool | How Framelets Enhance Graph Neural Networks | https://github.com/YuGuangWang/UFG | ICML 2021 | http://proceedings.mlr.press/v139/zheng21c.html |

#### Graph Scattering Transforms
| Number | GNN or method | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | | Diffusion Scattering Transforms on Graphs | | ICLR 2019 | https://openreview.net/forum?id=BygqBiRcFQ |
| 2 | | Stability of Graph Scattering Transforms | https://github.com/alelab-upenn/graph-scattering-transforms | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/3ce3bd7d63a2c9c81983cc8e9bd02ae5-Abstract.html |
| 3 | | Geometric Scattering for Graph Data Analysis | | ICML 2019 | http://proceedings.mlr.press/v97/gao19e.html |
| 4 | | Graph convolutional neural networks via scattering | https://github.com/dmzou/SCAT | Applied and Computational Harmonic Analysis | https://www.sciencedirect.com/science/article/abs/pii/S1063520318300678 |
| 5 | | Geometric Wavelet Scattering Networks on Compact Riemannian Manifolds | | ICML 2020 | https://proceedings.mlr.press/v107/perlmutter20a.html |
| 5 | | Data-Driven Learning of Geometric Scattering Networks | | | https://arxiv.org/abs/2010.02415 |
| 6 | | Understanding Graph Neural Networks with Asymmetric Geometric Scattering Transforms | | | https://arxiv.org/abs/1911.06253 |
| 7 | Scattering GCN | Scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks | https://github.com/dms-net/scatteringGCN | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/a6b964c0bb675116a15ef1325b01ff45-Abstract.html |
| 8 | GSAN | Geometric Scattering Attention Networks | https://github.com/dms-net/Attention-based-Scattering | ICASSP | https://ieeexplore.ieee.org/abstract/document/9414557/ |

#### Bayesian GNN
| Number | GNN | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | Bayesian GCN | Bayesian graph convolutional neural networks for semi-supervised classification | https://github.com/huawei-noah/BGCN | AAAI 2019 | https://ojs.aaai.org//index.php/AAAI/article/view/4531 |
| 2 | GPN | Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification | https://github.com/stadlmax/Graph-Posterior-Network | NeurIPS 2021 | https://papers.nips.cc/paper/2021/hash/95b431e51fc53692913da5263c214162-Abstract.html |

#### Graph Pooling (Graph Coarsening)
| Number | Graph Pooling | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | LaPool | Towards Interpretable Sparse Graph Representation Learning with Laplacian Pooling | | | https://arxiv.org/abs/1905.11577 |
| 2 | EigenPooling | Graph Convolutional Networks with EigenPooling | https://github.com/alge24/eigenpooling | KDD 2019 | https://dl.acm.org/doi/10.1145/3292500.3330982 |
| 3 | HaarPool | Haar Graph Pooling | https://github.com/YuGuangWang/HaarPool | ICML 2020 | http://proceedings.mlr.press/v119/wang20m.html |

### Spatial/Vertex domain GNNs
#### Convolutional GNNs
| Number | GNN | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | DCNN | Diffusion-Convolutional Neural Networks | https://github.com/RicardoZiTseng/dcnn-tensorflow | NeurIPS 2016 | https://proceedings.neurips.cc/paper/2016/hash/390e982518a50e280d8e2b535462ec1f-Abstract.html |
| 2 | GraphSAGE | Inductive Representation Learning on Large Graphs | https://github.com/williamleif/GraphSAGE | NeurIPS 2017 | https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html |
| 3 | FastGCN | FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling | https://github.com/matenure/FastGCN | ICLR 2018 | https://openreview.net/forum?id=rytstxWAW |
| 4 | JK-Net | Representation Learning on Graphs with Jumping Knowledge Networks | https://github.com/shinkyuy/representation_learning_on_graphs_with_jumping_knowledge_networks | ICML 2018 | http://proceedings.mlr.press/v80/xu18c.html |
| 5 | GIN | How Powerful are Graph Neural Networks? | https://github.com/weihua916/powerful-gnns | ICLR 2019 | https://openreview.net/forum?id=ryGs6iA5Km |
| 6 | k-GNN | Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks | https://github.com/chrsmrrs/k-gnn | AAAI 2019 | https://ojs.aaai.org/index.php/AAAI/article/view/4384 |
| 7 | Cluster-GCN | Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks | https://github.com/google-research/google-research/tree/master/cluster_gcn | KDD 2019 | https://dl.acm.org/doi/abs/10.1145/3292500.3330925 |
| 8 | Geom-GCN | Geom-GCN: Geometric Graph Convolutional Networks | https://github.com/graphdml-uiuc-jlu/geom-gcn | ICLR 2020 | https://openreview.net/forum?id=S1e2agrFvS |
| 9 | DAGNN | Towards Deeper Graph Neural Networks | https://github.com/divelab/DeeperGNN | KDD 2020 | https://dl.acm.org/doi/abs/10.1145/3394486.3403076 |
| 10 | H<sub>2</sub>GCN | Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs | https://github.com/GemsLab/H2GCN | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/58ae23d878a47004366189884c2f8440-Abstract.html |
| 11 | | Weisfeiler and Leman go sparse: Towards scalable higher-order graph embeddings | https://github.com/chrsmrrs/sparsewl | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/f81dee42585b3814de199b2e88757f5c-Abstract.html |
| 12 | GINE | Graph convolutions that can finally model local structure | https://github.com/RBrossard/GINEPLUS | | https://arxiv.org/abs/2011.15069 |
| 13 | DGN | Directional Graph Networks | https://github.com/Saro00/DGN | ICML 2021 | http://proceedings.mlr.press/v139/beaini21a.html |
| 14 | Elastic GNN | Elastic Graph Neural Networks | https://github.com/lxiaorui/ElasticGNN | ICML 2021 | https://proceedings.mlr.press/v139/liu21k.html |
| 15 | SIN | Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks | https://github.com/twitter-research/cwn | ICML 2021 | https://proceedings.mlr.press/v139/bodnar21a.html |
| 16 | | Don't stack layers in graph neural networks, wire them randomly | | ICLR 2021 Workshop | https://openreview.net/forum?id=xFH_wIFy1Je |
| 17 | CIN | Weisfeiler and Lehman Go Cellular: CW Networks | https://github.com/twitter-research/cwn | NeurIPS 2021 | https://openreview.net/forum?id=uVPZCMVtsSG |
| 18 | VQ-GNN | VQ-GNN: A Universal Framework to Scale-up Graph Neural Networks using Vector Quantization | https://github.com/devnkong/VQ-GNN | NeurIPS 2021 | https://openreview.net/forum?id=EO-CQzgcIxd |
| 19 | NDLS | Node Dependent Local Smoothing for Scalable Graph Learning | https://github.com/zwt233/ndls | NeurIPS 2021 | https://openreview.net/forum?id=ekKaTdleJVq |
| 20 | GraphSNN | A New Perspective on "How Graph Neural Networks Go Beyond Weisfeiler-Lehman?" | | ICLR 2022 | https://openreview.net/forum?id=uxgg9o7bI_3 |
| 21 | DS-GNN | Equivariant Subgraph Aggregation Networks | | ICLR 2022 | https://openreview.net/forum?id=dFbKQaRk15w |
| 22 | PEG | Equivariant and Stable Positional Encoding for More Powerful Graph Neural Networks  | https://github.com/graph-com/peg | ICLR 2022 | https://openreview.net/forum?id=e95i1IHcWj |
| 23 | CRaWl | Walking Out of the Weisfeiler Leman Hierarchy: Graph Learning Beyond Message Passing | https://github.com/toenshoff/CRaWl | TMLR 2023 | https://openreview.net/forum?id=vgXnEyeWVY |

#### Attentional GNNs
| Number | GNN | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | AGNN | Attention-based Graph Neural Network for Semi-supervised Learning | https://github.com/dawnranger/pytorch-AGNN | | https://arxiv.org/abs/1803.03735 |
| 2 | GAT | Graph Attention Network | https://github.com/PetarV-/GAT | ICLR 2018 | https://openreview.net/forum?id=rJXMpikCZ |
| 3 | MCN | Higher-order Graph Convolutional Networks | | | |
| 4 | CS-GNN | Measuring and Improving the Use of Graph Information in Graph Neural Networks | https://github.com/yifan-h/CS-GNN | ICLR 2020 | https://openreview.net/forum?id=rkeIIkHKvS |
| 5 | MixHop | MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing | https://github.com/samihaija/mixhop | ICML 2019 | http://proceedings.mlr.press/v97/abu-el-haija19a.html |
| 6 | GaAN | GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs | https://github.com/jennyzhang0215/GaAN | UAI 2018 | http://www.auai.org/uai2018/proceedings/papers/139.pdf |
| 7 | GAM | Graph Classification using Structural Attention | https://github.com/benedekrozemberczki/GAM | KDD 2018 | https://dl.acm.org/doi/abs/10.1145/3219819.3219980 |
| 8 | hGANet | Graph Representation Learning via Hard and Channel-Wise Attention Networks | | KDD 2019 | https://dl.acm.org/doi/abs/10.1145/3292500.3330897 |
| 9 | RGCN and RGAT | Relational Graph Attention Networks | https://github.com/babylonhealth/rgat | | https://arxiv.org/abs/1904.05811 |
| 10 | C-GAT | Improving Graph Attention Networks with Large Margin-based Constraints | | NeurIPS 2019 Workshop | https://grlearning.github.io/papers/43.pdf |
| 11 | FAGCN | Beyond Low-frequency Information in Graph Convolutional Networks | https://github.com/bdy9527/FAGCN | AAAI 2021 | https://ojs.aaai.org/index.php/AAAI/article/view/16514 |
| 12 | CAT-I and CAT-E | Learning Conjoint Attentions for Graph Neural Nets | | NeurIPS 2021 | https://openreview.net/forum?id=SMU_hbhhEQ |
| 13 | SuperGAT | How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision | https://github.com/dongkwan-kim/SuperGAT | ICLR 2021 | https://openreview.net/forum?id=Wi5KUNlqWty |
| 14 | GATv2 | How Attentive are Graph Attention Networks? | https://github.com/tech-srl/how_attentive_are_gats | ICLR 2022 | https://openreview.net/forum?id=F72ximsx7C1 |
| 15 | MLP-GAT | Graph Attention Retrospective | https://github.com/opallab/Graph-Attention-Retrospective | JMLR 2023 | https://jmlr.org/papers/v24/22-125.html |
| 16 | CAT and L-CAT | Learnable Graph Convolutional Attention Networks | https://github.com/psanch21/lcat | ICLR 2023 | https://openreview.net/forum?id=WsUMeHPo-2 |
| 17 | Dynamo-GAT | A Dynamical Systems-Inspired Pruning Strategy for Addressing Oversmoothing in Graph Attention Networks | | ICML 2025 | https://openreview.net/forum?id=44gnGhurnZ |

#### Graph Recurrent Neural Networks
| Number | GRNN | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | Graph LSTM and Gated Graph ConvNet | Residual Gated Graph ConvNets | https://github.com/xbresson/spatial_graph_convnets | ICLR 2018 | https://openreview.net/forum?id=HyXBcYg0b |

#### Graph Pooling (Graph Coarsening)
| Number | Graph Pooling | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | SortPooling | An End-to-End Deep Learning Architecture for Graph Classification | https://github.com/muhanzhang/DGCNN | AAAI 2018 | https://ojs.aaai.org/index.php/AAAI/article/view/11782 |
| 2 | DiffPool | Hierarchical Graph Representation Learning with Differentiable Pooling | https://github.com/RexYing/diffpool | NeurIPS 2018 | https://proceedings.neurips.cc/paper/2018/hash/e77dbaf6759253c7c6d0efc5690369c7-Abstract.html |
| 3 | gPool and gUnpool | Graph U-Nets | https://github.com/HongyangGao/Graph-U-Nets | ICML 2019 | http://proceedings.mlr.press/v97/gao19a.html |
| 4 | SAGPool | Self-Attention Graph Pooling | https://github.com/inyeoplee77/SAGPool | ICML 2019 | https://proceedings.mlr.press/v97/lee19c.html |
| 5 | Relational Pooling | Relational Pooling for Graph Representations | https://github.com/PurdueMINDS/RelationalPooling | ICML 2019 | http://proceedings.mlr.press/v97/murphy19a.html |
| 6 | HPL-SL | Hierarchical Graph Pooling with Structure Learning | https://github.com/cszhangzhen/HGP-SL | | https://arxiv.org/abs/1911.05954 |
| 7 | StructPool | StructPool: Structured Graph Pooling via Conditional Random Fields | https://github.com/Nate1874/StructPool | ICLR 2020 | https://openreview.net/forum?id=BJxg_hVtwH |
| 8 | MinCutPool | Spectral Clustering with Graph Neural Networks for Graph Pooling | https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling | ICML 2020 | https://proceedings.mlr.press/v119/bianchi20a.html |
| 9 | GSAPool | Structure-Feature based Graph Self-adaptive Pooling | | WWW 2020 | https://dl.acm.org/doi/10.1145/3366423.3380083 |
| 10 | NDP | Hierarchical Representation Learning in Graph Neural Networks with Node Decimation Pooling | https://github.com/danielegrattarola/decimation-pooling | IEEE Transactions on Neural Networks and Learning Systems | https://ieeexplore.ieee.org/abstract/document/9311759 |
| 11 | MxPool | MxPool: Multiplex Pooling for Hierarchical Graph Representation Learning | https://github.com/JucatL/MxPool/ | | https://arxiv.org/abs/2004.06846 |
| 12 | VIPool | Graph Cross Networks with Vertex Infomax Pooling | https://github.com/limaosen0/GXN | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/a26398dca6f47b49876cbaffbc9954f9-Abstract.html |
| 13 | GMT | Accurate Learning of Graph Representations with Graph Multiset Pooling | https://github.com/JinheonBaek/GMT | ICLR 2021 | https://openreview.net/forum?id=JHcqXGaqiGn |
| 14 | iPool | iPool—Information-Based Pooling in Hierarchical Graph Neural Networks | | IEEE Transactions on Neural Networks and Learning Systems | https://ieeexplore.ieee.org/document/9392315 |

#### MPNNs
| Number | MPNN | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | MPNN | Neural Message Passing for Quantum Chemistry | https://github.com/brain-research/mpnn | ICML 2017 | https://proceedings.mlr.press/v70/gilmer17a.html |
| 2 | GEN | DeeperGCN: All You Need to Train Deeper GCNs | https://github.com/lightaime/deep_gcns_torch | | https://arxiv.org/abs/2006.07739 |
| 3 | SMP | Building powerful and equivariant graph neural networks with structural message-passing | https://github.com/cvignac/SMP | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/a32d7eeaae19821fd9ce317f3ce952a7-Abstract.html |
| 4 | PNA | Principal Neighbourhood Aggregation for Graph Nets | https://github.com/lukecavabarrett/pna | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/99cad265a1768cc2dd013f0e740300ae-Abstract.html |
| 5 | GNNML | Breaking the Limits of Message Passing Graph Neural Networks | https://github.com/balcilar/gnn-matlang | ICML 2021 | https://proceedings.mlr.press/v139/balcilar21a.html |
| 6 | EGC | Do We Need Anisotropic Graph Neural Networks? | https://github.com/shyam196/egc | ICLR 2022 | https://openreview.net/forum?id=hl9ePdHO4_s |
| 7 | ACMP-GCN and ACMP-GAT | ACMP: Allen-Cahn Message Passing with Attractive and Repulsive Forces for Graph Neural Networks | https://github.com/ykiiiiii/acmp | ICLR 2023 | https://openreview.net/forum?id=4fZc_79Lrqs |
| 8 | CO-GNN | Cooperative Graph Neural Networks | https://github.com/benfinkelshtein/cognn | ICML 2024 | https://proceedings.mlr.press/v235/finkelshtein24a.html |

#### Heterogeneous Graph Neural Networks
| Number | HGNN | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | HetGNN | Heterogeneous Graph Neural Network | https://github.com/chuxuzhang/KDD2019_HetGNN | KDD 2019 | https://dl.acm.org/doi/10.1145/3292500.3330961 |
| 2 | HAN | Heterogeneous Graph Attention Network | https://github.com/Jhy1993/HAN | WWW 2019 | https://dl.acm.org/doi/10.1145/3292500.3330961 |
| 3 | MAGNN | MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding | https://github.com/cynricfu/MAGNN | WWW 2020 | https://dl.acm.org/doi/10.1145/3366423.3380297 |
| 4 | DGAT | Representation Learning on Heterophilic Graph with Directional Neighborhood Attention | | | https://arxiv.org/abs/2403.01475 |

#### Hyperbolic Graph Neural Networks
| Number | HGNN or method | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | HGCN | Hyperbolic Graph Convolutional Neural Networks | https://github.com/HazyResearch/hgcn | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/0415740eaa4d9decbc8da001d3fd805f-Abstract.html |
| 2 | HGNN | Hyperbolic Graph Neural Networks | https://github.com/facebookresearch/hgnn | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/103303dd56a731e377d01f6a37badae3-Abstract.html |
| 3 | HAT | Hyperbolic Graph Attention Network | | ICLR 2019 | https://openreview.net/forum?id=rJxHsjRqFQ |
| 4 | k-GCN | Constant Curvature Graph Convolutional Networks | | ICML 2020 | https://proceedings.mlr.press/v119/bachmann20a.html |
| 5 | HNN + HBN | Differentiating through the Fréchet Mean | https://github.com/CUAI/Differentiable-Frechet-Mean | ICML 2020 | http://proceedings.mlr.press/v119/lou20a.html |
| 6 | LGCN | Lorentzian Graph Convolutional Networks | | WWW 2021 | https://dl.acm.org/doi/abs/10.1145/3442381.3449872 |
| 7 | H2H-GCN | A Hyperbolic-to-Hyperbolic Graph Convolutional Network | | CVPR 2021 | https://openaccess.thecvf.com/content/CVPR2021/html/Dai_A_Hyperbolic-to-Hyperbolic_Graph_Convolutional_Network_CVPR_2021_paper.html |
| 8 | HGCF | HGCF: Hyperbolic Graph Convolution Networks for Collaborative Filtering | https://github.com/layer6ai-labs/HGCF | WWW 2021 | https://dl.acm.org/doi/10.1145/3442381.3450101 |
| 9 | HVGNN | Hyperbolic Variational Graph Neural Network for Modeling Dynamic Graphs | | AAAI 2021 | https://ojs.aaai.org/index.php/AAAI/article/view/16563 |
| 10 | ACE-HGNN | ACE-HGNN: Adaptive Curvature Exploration Hyperbolic Graph Neural Network | https://github.com/ringbdstack/ace-hgnn | | https://arxiv.org/abs/2110.07888 |

#### Capsule Graph Neural Network
| Number | CGNN or method | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | GCAPS-CNN | Graph Capsule Convolutional Neural Networks | https://github.com/vermaMachineLearning/Graph-Capsule-CNN-Networks/ | | https://arxiv.org/abs/1805.08090 |
| 2 | CapsGNN | Capsule Graph Neural Network | https://github.com/benedekrozemberczki/CapsGNN | ICLR 2019 | https://openreview.net/forum?id=Byl8BnRcYm |
| 3 | NCGNN | NCGNN: Node-level Capsule Graph Neural Network | | | https://arxiv.org/abs/2012.03476 |
| 4 | HGCN | Hierarchical Graph Capsule Network | https://github.com/uta-smile/HGCN | AAAI 2021 | https://ojs.aaai.org/index.php/AAAI/article/view/17268 |

#### Graph Neural ODE or PDE
| Number | GNODE or GNPDE or method | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | | Graph Neural Ordinary Differential Equations | https://github.com/Zymrael/gde | | https://arxiv.org/abs/1911.07532 |
| 2 | | Ordinary differential equations on graph networks | | | https://openreview.net/forum?id=SJg9z6VFDr |
| 3 | CGF | Continuous Graph Flow | | | https://arxiv.org/abs/1908.02436 |
| 4 | NODEC | Neural Ordinary Differential Equation Control of Dynamics on Graphs | https://github.com/asikist/nnc | IEEE DataPort 2020 | https://ieee-dataport.org/documents/neural-ordinary-differential-equation-control-dynamics-graphs |
| 5 | CGNN | Continuous Graph Neural Networks | https://github.com/DeepGraphLearning/ContinuousGNN | ICML 2020 | https://proceedings.mlr.press/v119/xhonneux20a.html |
| 6 | NDCN | Neural Dynamics on Complex Networks | https://github.com/calvin-zcx/ndcn | KDD 2020 | https://dl.acm.org/doi/abs/10.1145/3394486.3403132 |
| 7 | DeltaGN and OGN | Hamiltonian Graph Networks with ODE Integrators | | NeurIPS 2019 Workshop | https://ml4physicalsciences.github.io/2019/files/NeurIPS_ML4PS_2019_30.pdf |
| 8 | CFD-GCN | Combining Differentiable PDE Solvers and Graph Neural Networks for Fluid Flow Prediction | https://github.com/locuslab/cfd-gcn | ICML 2020 | https://proceedings.mlr.press/v119/de-avila-belbute-peres20a.html |
| 9 | GKN | Neural Operator: Graph Kernel Network for Partial Differential Equations | https://github.com/zongyi-li/graph-pde | ICLR 2020 Workshop | https://openreview.net/forum?id=fg2ZFmXFO3 |
| 10 | MGKN | Multipole Graph Neural Operator for Parametric Partial Differential Equations | https://github.com/zongyi-li/graph-pde | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/4b21cf96d4cf612f239a6c322b10c8fe-Abstract.html |
| 11 | | Learning continuous-time PDEs from sparse data with graph neural networks | | ICLR 2021 | https://openreview.net/forum?id=aUX5Plaq7Oy |
| 12 | GRAND | GRAND: Graph Neural Diffusion | https://github.com/twitter-research/graph-neural-pde | ICML 2021 | https://proceedings.mlr.press/v139/chamberlain21a.html |
| 13 | GraphCON-GCN and GraphCON-GAT | Graph-Coupled Oscillator Networks | https://github.com/tk-rusch/graphcon | ICML 2022 | https://proceedings.mlr.press/v162/rusch22a.html |
| 14 | G2-GCN and G2-GAT | Gradient Gating for Deep Multi-Rate Learning on Graphs | https://github.com/tk-rusch/gradientgating | ICLR 2023 | https://openreview.net/forum?id=JpRExTbl1- |
| 15 | GREAD | GREAD: Graph Neural Reaction-Diffusion Networks | https://github.com/jeongwhanchoi/GREAD | ICML 2023 | https://proceedings.mlr.press/v202/choi23a.html |

## List for Over-smoothing
### Analyses
| Number | Paper | Code | Journal or Conference | URL |
|:------:|--------------------------|-------|:-------:|----------------------------------------|
| 1 | Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning | https://github.com/liqimai/gcn | AAAI 2018 | https://ojs.aaai.org/index.php/AAAI/article/view/11604 |
| 2 | DeepGCNs: Can GCNs Go as Deep as CNNs? | https://github.com/lightaime/deep_gcns | ICCV 2019 | https://openaccess.thecvf.com/content_ICCV_2019/html/Li_DeepGCNs_Can_GCNs_Go_As_Deep_As_CNNs_ICCV_2019_paper.html |
| 3 | Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View | | AAAI 2020 | https://ojs.aaai.org//index.php/AAAI/article/view/5747 |
| 4 | Graph Neural Networks Exponentially Lose Expressive Power for Node Classification | https://github.com/delta2323/gnn-asymptotics | ICLR 2020 | https://openreview.net/forum?id=S1ldO2EFPr |
| 5 | A Note on Over-Smoothing for Graph Neural Networks | https://github.com/Chen-Cai-OSU/GNN-Over-Smoothing | ICML 2020 Workshop | https://arxiv.org/abs/2006.13318 |
| 6 | Revisiting Over-smoothing in Deep GCNs | | | https://arxiv.org/abs/2003.13663 |
| 7 | Measuring and Improving the Use of Graph Information in Graph Neural Networks | https://github.com/yifan-h/CS-GNN | ICLR 2020 | https://openreview.net/forum?id=rkeIIkHKvS |
| 8 | Simple and Deep Graph Convolutional Networks | https://github.com/chennnM/GCNII | ICML 2020 | https://proceedings.mlr.press/v119/chen20v.html |
| 9 | Graph Neural Networks with Adaptive Residual | https://github.com/lxiaorui/AirGNN | NeurIPS 2021 | https://proceedings.neurips.cc/paper/2021/hash/50abc3e730e36b387ca8e02c26dc0a22-Abstract.html |
| 10 | Two Sides of the Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks | | | https://arxiv.org/abs/2102.06462 |
| 11 | Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs | | | https://arxiv.org/abs/2202.04579 |
| 12 | Understanding convolution on graphs via energies | | | https://arxiv.org/abs/2206.10991 |
| 13 | A Non-Asymptotic Analysis of Oversmoothing in Graph Neural Networks | | ICLR 2023 | https://openreview.net/forum?id=CJd-BtnwtXq |
| 14 | A Fractional Graph Laplacian Approach to Oversmoothing | https://github.com/rpaolino/flode | NeurIPS 2023 | https://openreview.net/forum?id=kS7ED7eE74 |
| 15 | Demystifying Oversmoothing in Attention-Based Graph Neural Networks | | NeurIPS 2023 | https://openreview.net/forum?id=Kg65qieiuB |
| 16 | Rank Collapse Causes Over-Smoothing and Over-Correlation in Graph Neural Networks | https://github.com/roth-andreas/rank_collapse | LoG 2023 | https://proceedings.mlr.press/v231/roth24a.html |

### Graph Normalization
| Number | Norm | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | PairNorm | PairNorm: Tackling Oversmoothing in GNNs | https://github.com/LingxiaoShawn/PairNorm | ICLR 2020 | https://openreview.net/forum?id=rkecl1rtwB |
| 2 | NodeNorm | Understanding and Resolving Performance Degradation in Graph Convolutional Networks | https://github.com/miafei/NodeNorm | CIKM 2021 | https://dl.acm.org/doi/abs/10.1145/3459637.3482488 |
| 3 | DGN | Towards Deeper Graph Neural Networks with Differentiable Group Normalization | https://github.com/Kaixiong-Zhou/DGN | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/33dd6dba1d56e826aac1cbf23cdcca87-Abstract.html |
| 4 | GraphNorm | GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training | https://github.com/lsj2408/GraphNorm | ICML 2021 | http://proceedings.mlr.press/v139/cai21e.html |
| 5 | ContraNorm | ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond | https://github.com/PKU-ML/ContraNorm | ICLR 2023 | https://openreview.net/forum?id=SM7XkJouWHm |

### Dropout-like or Sampling
| Number | Method or GNN | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | DropEdge | DropEdge: Towards Deep Graph Convolutional Networks on Node Classification | https://github.com/DropEdge/DropEdge | ICLR 2020 | https://openreview.net/forum?id=Hkx1qkrKPr |
| 1 | DropEdge | Tackling Over-Smoothing for General Graph Convolutional Networks | https://github.com/DropEdge/DropEdge | IEEE Transactions on Pattern Analysis and Machine Intelligence | https://arxiv.org/abs/2008.09864 |
| 2 | FastGCN | FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling | https://github.com/matenure/FastGCN | ICLR 2018 | https://openreview.net/forum?id=rytstxWAW |
| 3 | VR-GCN | Stochastic Training of Graph Convolutional Networks with Variance Reduction | https://github.com/thu-ml/stochastic_gcn | ICML 2018 | https://proceedings.mlr.press/v80/chen18p.html |
| 4 | | Adaptive Sampling Towards Fast Graph Representation Learning | https://github.com/huangwb/AS-GCN | NeurIPS 2018 | https://proceedings.neurips.cc/paper/2018/hash/01eee509ee2f68dc6014898c309e86bf-Abstract.html |
| 5 | | Advancing GraphSAGE with A Data-driven Node Sampling | https://github.com/oj9040/GraphSAGE_RL | ICLR 2019 workshop | https://arxiv.org/abs/1904.12935 |
| 6 | LADIES | Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks | https://github.com/acbull/LADIES | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/91ba4a4478a66bee9812b0804b6f9d1b-Abstract.html |
| 7 | BBGDC | Bayesian Graph Neural Networks with Adaptive Connection Sampling | https://github.com/armanihm/GDC | ICML 2020 | https://proceedings.mlr.press/v119/hasanzadeh20a.html |
| 8 | GraphSAINT | GraphSAINT: Graph Sampling Based Inductive Learning Method | https://github.com/GraphSAINT/GraphSAINT | ICLR 2020 | https://openreview.net/forum?id=BJe8pkHFwS |
| 9 | MVS-GNN | Minimal Variance Sampling with Provable Guarantees for Fast Training of Graph Neural Networks | | KDD 2020 | https://dl.acm.org/doi/10.1145/3394486.3403192 |
| 10 | Critical DropEdge | Towards Deepening Graph Neural Networks: A GNTK-based Optimization Perspective | | ICLR 2022 | https://openreview.net/forum?id=tT9t_ZctZRL |

## List for Over-squashing
| Number | Paper | Code | Journal or Conference | URL |
|:------:|--------------------------|-------|:-------:|----------------------------------------|
| 1 | On the Bottleneck of Graph Neural Networks and its Practical Implications | https://github.com/tech-srl/bottleneck/ | ICLR 2021 | https://openreview.net/forum?id=i80OPhOCVH2 |
| 2 | Understanding over-squashing and bottlenecks on graphs via curvature | | ICLR 2022 | https://openreview.net/forum?id=7UmjRGzp-A |
| 3 | On Over-Squashing in Message Passing Neural Networks: The Impact of Width, Depth, and Topology | | ICML 2023 | https://proceedings.mlr.press/v202/di-giovanni23a.html |
| 4 | How does over-squashing affect the power of GNNs? | | TMLR 2024 | https://openreview.net/forum?id=KJRoQvRWNs |
| 5 | PANDA: Expanded Width-Aware Message Passing Beyond Rewiring | | ICML 2024 | https://proceedings.mlr.press/v235/choi24f.html |

## List for Graph Transformers
| Number |  Graph Transformer | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | Graph-Bert | Graph-Bert: Only Attention is Needed for Learning Graph Representations | https://github.com/jwzhanggy/Graph-Bert | | https://arxiv.org/abs/2001.05140 |
| 2 | GTN | Graph Transformer Networks | https://github.com/seongjunyun/Graph_Transformer_Networks | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/9d63484abb477c97640154d40595a3bb-Abstract.html |
| 3 | HGT | Heterogeneous Graph Transformer | https://github.com/acbull/pyHGT | WWW 2020 | https://dl.acm.org/doi/10.1145/3366423.3380027 |
| 4 | GT | A Generalization of Transformer Networks to Graphs | https://github.com/graphdeeplearning/graphtransformer | AAAI 2021 Workshop | https://arxiv.org/abs/2012.09699 |
| 5 | UniMP | Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification | https://github.com/PaddlePaddle/PGL/tree/main/ogb_examples/nodeproppred/unimp | IJCAI 2021 | https://www.ijcai.org/proceedings/2021/214 |
| 6 | GraphTrans | Representing Long-Range Context for Graph Neural Networks with Global Attention | https://github.com/ucbrise/graphtrans | NeurIPS 2021 | https://proceedings.neurips.cc/paper/2021/hash/6e67691b60ed3e4a55935261314dd534-Abstract.html |
| 7 | SAN | Rethinking Graph Transformers with Spectral Attention | https://github.com/DevinKreuzer/SAN | NeurIPS 2021 | https://openreview.net/forum?id=huAdB-Tj4yG |
| 8 | Graphormer | Do Transformers Really Perform Badly for Graph Representation? | https://github.com/Microsoft/Graphormer | NeurIPS 2021 | https://openreview.net/forum?id=OeWooOxFwDa |
| 9 | Graphormer | Benchmarking Graphormer on Large-Scale Molecular Modeling Datasets | https://github.com/Microsoft/Graphormer | | https://arxiv.org/abs/2203.04810 |
| 10 | Gophormer | Gophormer: Ego-Graph Transformer for Node Classification | | | https://arxiv.org/abs/2110.13094 |
| 11 | SEA | SEA: Graph Shell Attention in Graph Neural Networks | | ECML/PKDD 2021 | https://arxiv.org/pdf/2110.10674.pdf |
| 12 | GraphiT | GraphiT: Encoding Graph Structure in Transformers | https://github.com/inria-thoth/GraphiT | | https://arxiv.org/abs/2106.05667 |
| 13 | Coarformer | Coarformer: Transformer for large graph via graph coarsening | | | https://openreview.net/forum?id=fkjO_FKVzw |
| 14 | UGformer | Universal Graph Transformer Self-Attention Networks | https://github.com/daiquocnguyen/Graph-Transformer | WWW 2022 | https://dl.acm.org/doi/10.1145/3487553.3524258 |
| 15 | EGT | Global Self-Attention as a Replacement for Graph Convolution | https://github.com/shamim-hussain/egt_pytorch | KDD 2022 | https://dl.acm.org/doi/10.1145/3534678.3539296 |
| 16 | GKAT | From block-Toeplitz matrices to differential equations on graphs: towards a general theory for scalable masked Transformers | https://github.com/hl-hanlin/gkat | ICML 2022 | https://arxiv.org/abs/2107.07999 |
| 17 | FeTA | Investigating Expressiveness of Transformer in Spectral Domain for Graphs | https://github.com/ansonb/FeTA_TMLR | TMLR 2022 | https://openreview.net/forum?id=aRsLetumx1 |
| 18 | TokenGT | Pure Transformers are Powerful Graph Learners | https://github.com/jw9730/tokengt | NeurIPS 2022 | https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d84236751fe6d25dc06db055a3180b0-Abstract-Conference.html |
| 19 | GPS | Recipe for a General, Powerful, Scalable Graph Transformer | https://github.com/rampasek/GraphGPS | NeurIPS 2022 | https://openreview.net/forum?id=lMMaNf6oxKM |
| 20 | GRPE | GRPE: Relative Positional Encoding for Graph Transformer | https://github.com/lenscloth/grpe | | https://arxiv.org/abs/2201.12787 |
| 21 | DGT | Deformable Graph Transformer | | | https://arxiv.org/abs/2206.14337 |
| 22 | Specformer | Specformer: Spectral Graph Neural Networks Meet Transformers | https://github.com/DSL-Lab/Specformer | ICLR 2023 | https://openreview.net/forum?id=0pdSt3oyJa1 |
| 23 | NAGphormer | NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs | https://github.com/JHL-HUST/NAGphormer | ICLR 2023 | https://openreview.net/forum?id=8KYeilT3Ow |
| 25 | DeepGraph | Are More Layers Beneficial to Graph Transformers? | https://github.com/zhao-ht/deepgraph | ICLR 2023 | https://openreview.net/forum?id=uagC-X9XMi8 |
| 26 | GRIT | Graph Inductive Biases in Transformers without Message Passing | https://github.com/liamma/grit | ICML 2023 | https://proceedings.mlr.press/v202/ma23c |
| 27 | Exphormer | Exphormer: Sparse Transformers for Graphs | https://github.com/hamed1375/Exphormer | ICML 2023 | https://proceedings.mlr.press/v202/shirzad23a.html |
| 28 | GOAT | GOAT: A Global Transformer on Large-scale Graphs | https://github.com/devnkong/GOAT | ICML 2023 | https://proceedings.mlr.press/v202/kong23a.html |
| 29 | RWC | Random Walk Conformer: Learning Graph Representation from Long and Short Range | https://github.com/b05901024/RandomWalkConformer | AAAI 2023 | https://ojs.aaai.org/index.php/AAAI/article/view/26296 |
| 30 | Gapformer | Gapformer: Graph Transformer with Graph Pooling for Node Classification | | IJCAI 2023 | https://www.ijcai.org/proceedings/2023/244 |
| 31 | UGT | Transitivity-Preserving Graph Representation Learning for Bridging Local Connectivity and Role-based Similarity | https://github.com/NSLab-CUK/Unified-Graph-Transformer | AAAI 2024 | https://ojs.aaai.org/index.php/AAAI/article/view/29138 |

## List for Graph Structure Embedding
| Number |  Graph Transformer | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | | Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting | https://github.com/gbouritsas/GSN | IEEE Transactions on Pattern Analysis and Machine Intelligence | https://ieeexplore.ieee.org/document/9721082 |
| 2 | | Rethinking Graph Transformers with Spectral Attention | https://openreview.net/forum?id=huAdB-Tj4yG | NeurIPS 2021 | https://github.com/DevinKreuzer/SAN | 
| 3 | LSPE | Graph Neural Networks with Learnable Structural and Positional Representations | https://github.com/vijaydwivedi75/gnn-lspe | ICLR 2022 | https://openreview.net/forum?id=wTTjnvGphYj | 
| 4 | | Structure-Aware Transformer for Graph Representation Learning | https://github.com/borgwardtlab/sat | ICML 2022 | https://proceedings.mlr.press/v162/chen22r.html |

## List for Graph State Space Models (GSSMs)
| Number |  GSSM | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | GRED | Recurrent Distance Filtering for Graph Representation Learning | | | https://arxiv.org/abs/2312.01538 |
| 2 | Graph-Mamba | Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces | https://github.com/bowang-lab/Graph-Mamba | | https://arxiv.org/abs/2402.00789 |
| 3 | Graph Mamba | Graph Mamba: Towards Learning on Graphs with State Space Models | https://github.com/graphmamba/gmn | | https://arxiv.org/abs/2402.08678 |

## List for Graph MLP
| Number | Graph MLP | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | Graph-MLP | Graph-MLP: Node Classification without Message Passing in Graph | https://github.com/yanghu819/Graph-MLP | | https://arxiv.org/abs/2106.04051 |
| 2 | N2N | Node Representation Learning in Graph via Node-to-Neighbourhood Mutual Information Maximization | https://github.com/dongwei156/n2n | | https://arxiv.org/abs/2203.12265 |

## List for Tensor Graph Neural Networks
| Number | TGNN | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | tGNN | High-Order Pooling for Graph Neural Networks with Tensor Decomposition | | NeurIPS 2022 | https://openreview.net/forum?id=N7-EIciq3R |
| 2 | TGCN | Efficient Relation-aware Neighborhood Aggregation in Graph Neural Networks via Tensor Decomposition | | | https://arxiv.org/abs/2212.05581 |
| 3 | THNN | Tensorized Hypergraph Neural Networks | | | https://arxiv.org/abs/2306.02560 |

## List for Graph Autoencoders (GAE)
| Number | GAE | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | DNGR | Deep Neural Networks for Learning Graph Representations | | AAAI 2016 | https://ojs.aaai.org/index.php/AAAI/article/view/10179 |
| 2 | SDNE | Structural Deep Network Embedding | | | |
| 3 | DVNE | Deep Variational Network Embedding in Wasserstein Space | | KDD 2016 | https://www.kdd.org/kdd2016/subtopic/view/structural-deep-network-embedding |
| 4 | VGAE | Variational Graph Auto-Encoders | https://github.com/tkipf/gae | | | https://arxiv.org/abs/1611.07308 |
| 5 | GC-MC | Graph Convolutional Matrix Completion | https://github.com/riannevdberg/gc-mc | | https://arxiv.org/abs/1706.02263 |
| 6 | ARVGA | Adversarially regularized graph autoencoder for graph embedding | | IJCAI 2018 | https://dl.acm.org/doi/10.5555/3304889.3305023 |
| 7 | NetRA | Learning deep network representations with adversarially regularized autoencoders | | KDD 2018 | https://dl.acm.org/doi/10.1145/3219819.3220000 |
| 8 | DeepGMG | Learning deep generative models of graphs | | | https://arxiv.org/abs/1803.03324 |
| 9 | GraphRNN | GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models | https://github.com/JiaxuanYou/graph-generation | ICML 2018 | https://proceedings.mlr.press/v80/you18a.html |
| 10 | GraphVAE | Graphvae: Towards generation of small graphs using variational autoencoders | | ICANN 2018 | https://link.springer.com/chapter/10.1007/978-3-030-01418-6_41 |
| 11 | | Constrained generation of semantically valid graphs via regularizing variational autoencoders | | NeurISP 2018 | https://proceedings.neurips.cc/paper/2018/hash/1458e7509aa5f47ecfb92536e7dd1dc7-Abstract.html |
| 12 | Gravity Graph VAE and Gravity Graph AE | Gravity-Inspired Graph Autoencoders for Directed Link Prediction | https://github.com/deezer/gravity_graph_autoencoders | CIKM 2019 | https://dl.acm.org/doi/abs/10.1145/3357384.3358023 |

## List for Graph Generative Adversarial Networks (GGAN)
| Number | GGAN | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | GraphGAN | GraphGAN: Graph Representation Learning with Generative Adversarial Nets | https://github.com/hwwang55/GraphGAN | AAAI 2018 | https://ojs.aaai.org/index.php/AAAI/article/view/11872 |
| 2 | MolGAN | MolGAN: An implicit generative model for small molecular graphs | | | https://arxiv.org/abs/1805.11973 |
| 3 | NetGAN | NetGAN: Generating graphs via random walks | | ICML 2018 | http://proceedings.mlr.press/v80/bojchevski18a.html |

## List for graph pre-training
| Number | Pre-training mathod | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | | Strategies for Pre-training Graph Neural Networks | https://github.com/snap-stanford/pretrain-gnns/ | ICLR 2020 | https://openreview.net/forum?id=HJlWWJSFDH |
| 2 | GCC | GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training | https://github.com/THUDM/GCC | KDD 2020 | https://dl.acm.org/doi/10.1145/3394486.3403168 |

## List for GNN explainers
| Number | GNN or method | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | GNNExplainer | GNNExplainer: Generating Explanations for Graph Neural Networks | https://github.com/RexYing/gnn-model-explainer | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html |
| 2 | PGExplainer | Parameterized Explainer for Graph Neural Network | https://github.com/flyingdoog/PGExplainer | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html |
| 3 | PGM-Explainer | PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks | https://github.com/vunhatminh/PGMExplainer | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/8fb134f258b1f7865a6ab2d935a897c9-Abstract.html |
| 4 | XGNN | XGNN: Towards Model-Level Explanations of Graph Neural Networks | | KDD 2020 | https://www.kdd.org/kdd2020/accepted-papers/view/xgnn-towards-model-level-explanations-of-graph-neural-networks |
| 5 | Gem | Generative Causal Explanations for Graph Neural Networks | https://github.com/wanyu-lin/ICML2021-Gem | ICML 2021 | https://proceedings.mlr.press/v139/lin21d.html |
| 6 | | When Comparing to Ground Truth is Wrong: On Evaluating GNN Explanation Methods | | KDD 2021 | https://dl.acm.org/doi/abs/10.1145/3447548.3467283 |
| 7 | MotifExplainer | MotifExplainer: a Motif-based Graph Neural Network Explainer | | | https://arxiv.org/abs/2202.00519 |

## List for Graph Adversarial Attacks and Defenses
| Number | method | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | | Adversarial Attacks on Neural Networks for Graph Data | https://github.com/danielzuegner/nettack | KDD 2018 | https://dl.acm.org/doi/10.1145/3219819.3220078 |
| 2 | | Certifiable Robustness and Robust Training for Graph Convolutional Networks | https://github.com/danielzuegner/robust-gcn | KDD 2020 | https://dl.acm.org/doi/abs/10.1145/3394486.3403217 |
| 3 | | Adversarial Attacks on Graph Neural Networks via Meta Learning | | ICLR 2019 | https://openreview.net/forum?id=Bylnx209YX |
| 4 | | Adversarial Attacks on Node Embeddings via Graph Poisoning | https://github.com/abojchevski/node_embedding_attack | ICML 2019 | https://proceedings.mlr.press/v97/bojchevski19a.html |
| 5 | GNNGuard | GNNGuard: Defending Graph Neural Networks against Adversarial Attacks | https://github.com/mims-harvard/GNNGuard | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/690d83983a63aa1818423fd6edd3bfdb-Abstract.html |
| 6 | | Detection and Defense of Topological Adversarial Attacks on Graphs | | ICML 2021 | https://proceedings.mlr.press/v130/zhang21i.html |
| 7 | GCN-LFR | Not All Low-Pass Filters are Robust in Graph Convolutional Networks | | NeurIPS 2021 | https://openreview.net/forum?id=bDdfxLQITtu |

## List for WL Test
| Number | method/model | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | GIN | How Powerful are Graph Neural Networks? | https://github.com/weihua916/powerful-gnns | ICLR 2019 | https://openreview.net/forum?id=ryGs6iA5Km |
| 2 | GraphSNN | A New Perspective on "How Graph Neural Networks Go Beyond Weisfeiler-Lehman?"  | https://github.com/wokas36/GraphSNN | ICLR 2022 | https://openreview.net/forum?id=uxgg9o7bI_3 |
| 3 | SEG-WL test | On Structural Expressive Power of Graph Transformers | | KDD 2023 | https://dl.acm.org/doi/10.1145/3580305.3599451 |
| 4 | N-WL test | N-WL: A New Hierarchy of Expressivity for Graph Neural Networks | | ICLR 2023 | https://openreview.net/forum?id=5cAI0qXxyv |
| 5 | GD-WL | Rethinking the Expressive Power of GNNs via Graph Biconnectivity | https://github.com/lsj2408/graphormer-gd | ICLR 2023 | https://openreview.net/forum?id=r9hNv76KoT3 |

## List for Others
| Number | GNN or method | Paper | Code | Journal or Conference | URL |
|:------:|:--------------------------:|-------|------|:-------:|----------------------------------------|
| 1 | | Contrastive Multi-View Representation Learning on Graphs | https://github.com/kavehhassani/mvgrl | ICML 2020 | https://proceedings.mlr.press/v119/hassani20a.html |
| 2 | FLAG | Robust Optimization as Data Augmentation for Large-scale Graphs | https://github.com/devnkong/FLAG | | https://arxiv.org/abs/2010.09891 |
| 3 | | Interpreting and Unifying Graph Neural Networks with An Optimization Framework | | WWW 2021 | https://dl.acm.org/doi/10.1145/3442381.3449953 |
| 4 | | What graph neural networks cannot learn: depth vs width | | ICLR 2020 | https://openreview.net/forum?id=B1l2bp4YwS |
| 5 | | Extract the Knowledge of Graph Neural Networks and Go Beyond it: An Effective Knowledge Distillation Framework | https://github.com/BUPT-GAMMA/CPF | WWW 2021 | https://dl.acm.org/doi/abs/10.1145/3442381.3450068 |
| 6 | SUGAR | SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism | https://github.com/RingBDStack/SUGAR | WWW 2021 | https://dl.acm.org/doi/10.1145/3442381.3449822 |
| 7 | | Towards Sparse Hierarchical Graph Classifiers | | NeurIPS 2018 Workshop | https://arxiv.org/abs/1811.01287 |
| 8 | OGB | Open Graph Benchmark: Datasets for Machine Learning on Graphs | https://github.com/snap-stanford/ogb | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html |
| 9 | AdaGCN | AdaGCN: Adaboosting Graph Convolutional Networks into Deep Models | https://github.com/datake/AdaGCN | ICLR 2021 | https://openreview.net/forum?id=QkRbdiiEjM |
| 10 | BGNN | Bilinear Graph Neural Network with Neighbor Interactions | https://github.com/zhuhm1996/bgnn | IJCAI 2020 | https://www.ijcai.org/proceedings/2020/202 |
| 11 | RevGNN-Deep and RevGNN-Wide | Training Graph Neural Networks with 1000 Layers | https://github.com/lightaime/deep_gcns_torch/tree/master/examples/ogb_eff/ogbn_proteins | ICML 2021 | https://proceedings.mlr.press/v139/li21o.html |
| 12 | OGB-LSC | OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs | | | https://arxiv.org/abs/2103.09430 |
| 13 | DrGCNs | Dimensional Reweighting Graph Convolutional Networks | | | https://arxiv.org/abs/1907.02237 |
| 14 | GAS | GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings | https://github.com/rusty1s/pyg_autoscale | ICML 2021 | http://proceedings.mlr.press/v139/fey21a.html |
| 15 | TWIRLS | Graph Neural Networks Inspired by Classical Iterative Algorithms | https://github.com/FFTYYY/TWIRLS | ICML 2021 | http://proceedings.mlr.press/v139/yang21g.html |
| 16 | GAT-Lip | Lipschitz Normalization for Self-Attention Layers with Application to Graph Neural Networks | | ICML 2021 | http://proceedings.mlr.press/v139/dasoulas21a.html |
| 17 | | Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective | https://github.com/balcilar/gnn-spectral-expressive-power | ICLR 2021 | https://openreview.net/forum?id=-qh0M9XWxnv |
| 18 | | Deep Graph Neural Networks with Shallow Subgraph Samplers | https://github.com/facebookresearch/shaDow_GNN | | https://arxiv.org/abs/2012.01380 |
| 19 | | Large-scale graph representation learning with very deep GNNs and self-supervision | https://github.com/deepmind/deepmind-research/tree/master/ogb_lsc | | https://arxiv.org/abs/2107.09422 |
| 20 | GCN-LPA | Unifying Graph Convolutional Neural Networks and Label Propagation | https://github.com/hwwang55/GCN-LPA | | Unifying Graph Convolutional Neural Networks and Label Propagation |
| 21 | L-GCN and L<sup>2</sup>-GCN | L<sup>2</sup>-GCN: Layer-Wise and Learned Efficient Training of Graph Convolutional Networks | https://github.com/VITA-Group/L2-GCN | CVPR 2020 | https://openaccess.thecvf.com/content_CVPR_2020/html/You_L2-GCN_Layer-Wise_and_Learned_Efficient_Training_of_Graph_Convolutional_Networks_CVPR_2020_paper.html |
| 22 | | A Fair Comparison of Graph Neural Networks for Graph Classification | https://github.com/diningphil/gnn-comparison | ICLR 2020 | https://openreview.net/forum?id=HygDF6NFPB |
| 23 | CurvGN | Curvature Graph Network | | ICLR 2020 | https://openreview.net/forum?id=BylEqnVFDB |
| 24 | GIB | Graph Information Bottleneck | https://github.com/snap-stanford/GIB | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/ebc2aa04e75e3caabda543a1317160c0-Abstract.html |
| 25 | ResRGAT | Improving Breadth-Wise Backpropagation in Graph Neural Networks Helps Learning Long-Range Dependencies | https://github.com/lukovnikov/resrgat | ICML 2021 | https://proceedings.mlr.press/v139/lukovnikov21a.html |
| 26 | | Optimization of Graph Neural Networks: Implicit Acceleration by Skip Connections and More Depth | | ICML 2021 | https://proceedings.mlr.press/v139/xu21k.html |
| 27 | MinGE | Graph Entropy Guided Node Embedding Dimension Selection for Graph Neural Networks | https://github.com/RingBDStack/MinGE | IJCAI 2021 | https://www.ijcai.org/proceedings/2021/381 |
| 28 | | A Flexible Generative Framework for Graph-based Semi-supervised Learning | https://github.com/jiaqima/G3NN | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/e0ab531ec312161511493b002f9be2ee-Abstract.html |
| 29 | GRAND | Graph Random Neural Networks for Semi-Supervised Learning on Graphs | https://github.com/THUDM/GRAND | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/fb4c835feb0a65cc39739320d7a51c02-Abstract.html |
| 30 | | Approximation Ratios of Graph Neural Networks for Combinatorial Problems | | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/635440afdfc39fe37995fed127d7df4f-Abstract.html |
| 31 | | Can Graph Neural Networks Count Substructures? | https://github.com/leichen2018/GNN-Substructure-Counting | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/75877cb75154206c4e65e76b88a12712-Abstract.html |
| 32 | GNN-FiLM | GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation | https://github.com/microsoft/tf-gnn-samples | ICML 2020 | https://proceedings.mlr.press/v119/brockschmidt20a.html |
| 33 | | Graph Attention Retrospective | | | https://arxiv.org/abs/2202.13060 |
| 34 | | GraphWorld: Fake Graphs Bring Real Insights for GNNs | | | https://arxiv.org/abs/2203.00112 |
| 35 | GRAND+ | GRAND+: Scalable Graph Random Neural Networks | https://github.com/THUDM/GRAND-plus | | https://arxiv.org/abs/2203.06389 |
| 36 | SAGN | Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training | https://github.com/skepsun/SAGN_with_SLE | | https://arxiv.org/abs/2104.09376 |
| 37 | GAMLP | Graph Attention Multi-Layer Perceptron | https://github.com/pku-dair/gamlp | KDD 2022 | https://dl.acm.org/doi/abs/10.1145/3534678.3539121 |
| 38 | | Expressiveness and Approximation Properties of Graph Neural Networks | | ICLR 2022 | https://openreview.net/forum?id=wIzUeM3TAU |
| 39 | | Towards Understanding Generalization of Graph Neural Networks | | ICML 2023 | https://proceedings.mlr.press/v202/tang23f.html |

## List for Surveys
| Number | Paper | Journal or Conference | URL |
|:------:|--------------------------|:-------:|----------------------------------------|
| 1 | Graph Neural Networks: A Review of Methods and Applications | AI Open | https://www.sciencedirect.com/science/article/pii/S2666651021000012 |
| 2 | A Comprehensive Survey on Graph Neural Networks | IEEE Transactions on Neural Networks and Learning Systems | https://ieeexplore.ieee.org/abstract/document/9046288 |
| 3 | Deep Learning on Graphs: A Survey | IEEE Transactions on Knowledge and Data Engineering | https://ieeexplore.ieee.org/abstract/document/9039675 |
| 4 | Explainability in Graph Neural Networks: A Taxonomic Survey | IEEE Transactions on Pattern Analysis and Machine Intelligence | https://www.computer.org/csdl/journal/tp/2023/05/09875989/1GqajxgkWcM |
| 5 | Graph Neural Networks for Natural Language Processing: A Survey | Foundations and Trends® in Machine Learning | https://www.nowpublishers.com/article/Details/MAL-096 |
| 6 | Benchmarking Graph Neural Networks | JMLR 2022 | https://www.jmlr.org/papers/v24/22-0567.html |
| 7 | Bridging the Gap between Spatial and Spectral Domains: A Unified Framework for Graph Neural Networks | ACM Computing Surveys | https://dl.acm.org/doi/10.1145/3627816 |
| 8 | Transformer for Graphs: An Overview from Architecture Perspective | | https://arxiv.org/abs/2202.08455 |
| 9 | Weisfeiler and Leman go Machine Learning: The Story so far | JMLR 2023 | https://jmlr.org/papers/v24/22-0240.html |
| 10 | Attending to Graph Transformers | TMLR 2024 | https://openreview.net/forum?id=HhbqHBBrfZ |
| 11 | The Heterophilic Graph Learning Handbook: Benchmarks, Models, Theoretical Analysis, Applications and Challenges | | https://arxiv.org/abs/2407.09618 |
