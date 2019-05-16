# Net2Net in Knet

This is the repo for my 2019 Deep Learning term project. This project consists of a replication of the paper [Net2Net: Accelerating Learning via Knowledge Transfer](https://arxiv.org/abs/1511.05641) using [Julia](https://julialang.org/) and [Knet](https://github.com/denizyuret/Knet.jl). 

Net2Net is a set of methods for growing a wider and/or deeper 'student' network from a trained 'teacher' network while preserving its function. The student network performs immediately as well as the teacher while possessing greater capacity that can be utilized by further training. Training a Net2Net initialized network is shown to converge faster than training from scratch.

While the original paper ran experiments using the Inception-BN network on the ImageNet dataset, this work uses a modified, smaller version of the network on the CIFAR-10 dataset. However, the original Inception-BN architecture is available in `inception.jl`, and can be used for experiments.

Check the following links for additional information:

- [Technical Report](https://www.overleaf.com/read/wxvptvtnrsdn)
- [Presentation](https://docs.google.com/presentation/d/1wHOqNkWw5V4LTdpCOc08QuG39zgqGQSDW-Vj3wkNl8M/edit?usp=sharing)
- [Data sheet with results](https://docs.google.com/spreadsheets/d/1mkuw2OMh9RdHeVrFEJfgqll0yJfqjLKJ9WGNHuO8nMg/edit?usp=sharing)
- [Research log](https://docs.google.com/spreadsheets/d/1mkuw2OMh9RdHeVrFEJfgqll0yJfqjLKJ9WGNHuO8nMg/edit?usp=sharing)
- [Original paper](https://arxiv.org/abs/1511.05641)

## Usage

To run the tests and experiments, simply run `source.jl`.

To use the Net2WiderNet or Net2DeeperNet operations, call the `wider_*` and `deeper_*` functions defined in `wider.jl` and `deeper.jl`. Refer to the docstrings for the usage of those functions.

`models.jl` and `inception.jl` files define strucures and functions for building models that can be used with the Net2WiderNet and Net2DeeperNet functions.
