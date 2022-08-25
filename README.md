# NBFNet: Neural Bellman-Ford Networks #

PyG re-implementation of [NBFNet]. Authored by [Zhaocheng Zhu] and [Michael Galkin].

[Zhaocheng Zhu]: https://kiddozhu.github.io
[Michael Galkin]: https://migalkin.github.io/
[NBFNet]: https://github.com/DeepGraphLearning/NBFNet

## Overview ##

NBFNet is a graph neural network framework inspired by traditional path-based
methods. It enjoys the advantages of both traditional path-based methods and modern
graph neural networks, including **generalization in the inductive setting**,
**interpretability**, **high model capacity** and **scalability**. This repo implements
NBFNet for transductive and inductive knowledge graph reasoning.

![NBFNet](asset/nbfnet.svg)

This codebase is based on PyTorch and PyTorch-Geometric. It supports training and
inference with multiple GPUs or multiple machines.

## Installation ##

You may install the dependencies via either conda or pip. Generally, NBFNet works
with Python >= 3.7 and PyTorch >= 1.8.0.

### From Conda ###

```bash
conda install pytorch=1.8.0 cudatoolkit=11.1 pyg -c pytorch -c pyg -c conda-forge
conda install ninja easydict pyyaml -c conda-forge
```

### From Pip ###

```bash
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.8 torch-sparse==0.6.12 torch-geometric -f https://data.pyg.org/whl/torch-1.8.0+cu111.html
pip install ninja easydict pyyaml
```

## Reproduction ##

To reproduce the results of NBFNet, use the following command. Alternatively, you
may use `--gpus null` to run NBFNet on a CPU. All the datasets will be automatically
downloaded in the code.

```bash
python script/run.py -c config/inductive/wn18rr.yaml --gpus [0] --version v1
```

We provide the hyperparameters for each experiment in configuration files.
All the configuration files can be found in `config/*/*.yaml`.

For experiments on inductive relation prediction, you need to additionally specify
the split version with `--version v1`.

To run NBFNet with multiple GPUs or multiple machines, use the following commands

```bash
python -m torch.distributed.launch --nproc_per_node=4 script/run.py -c config/inductive/wn18rr.yaml --gpus [0,1,2,3]
```

```bash
python -m torch.distributed.launch --nnodes=4 --nproc_per_node=4 script/run.py -c config/inductive/wn18rr.yaml --gpus [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
```

### Visualize Interpretations on FB15k-237 ###

Once you have models trained on FB15k237, you can visualize the path interpretations
with the following line. Please replace the checkpoint with your own path.

```bash
python script/visualize.py -c config/transductive/fb15k237_visualize.yaml --checkpoint /path/to/nbfnet/experiment/model_epoch_20.pth
```

## Results ##

Here are the benchmark results of this re-implementation. All the results are
obtained with 4 V100 GPUs (32GB). Note results may be slightly different if the
model is trained with 1 GPU and/or a smaller batch size.

### Knowledge Graph Completion ###

<table>
    <tr>
        <th>Dataset</th>
        <th>MR</th>
        <th>MRR</th>
        <th>HITS@1</th>
        <th>HITS@3</th>
        <th>HITS@10</th>
    </tr>
    <tr>
        <th>FB15k-237</th>
        <td>113</td>
        <td>0.416</td>
        <td>0.322</td>
        <td>0.456</td>
        <td>0.602</td>
    </tr>
    <tr>
        <th>WN18RR</th>
        <td>650</td>
        <td>0.551</td>
        <td>0.496</td>
        <td>0.573</td>
        <td>0.661</td>
    </tr>
</table>

<table>
    <tr>
        <th>Dataset</th>
        <th>Training Time</th>
        <th>Test Time</th>
        <th>GPU Memory</th>
    </tr>
    <tr>
        <th>FB15k-237</th>
        <td>22.8 min / epoch</td>
        <td>64 s</td>
        <td>22.9 GiB</td>
    </tr>
    <tr>
        <th>WN18RR</th>
        <td>12.0 min / epoch</td>
        <td>16 s</td>
        <td>16.1 GiB</td>
    </tr>
</table>

### Inductive Relation Prediction ###

<table>
    <tr>
        <th rowspan="2">Dataset</th>
        <th colspan="4">HITS@10 (50 sample)</th>
    </tr>
    <tr>
        <th>v1</th>
        <th>v2</th>
        <th>v3</th>
        <th>v4</th>
    </tr>
    <tr>
        <th>FB15k-237</th>
        <td>0.821</td>
        <td>0.948</td>
        <td>0.957</td>
        <td>0.959</td>
    </tr>
    <tr>
        <th>WN18RR</th>
        <td>0.954</td>
        <td>0.903</td>
        <td>0.909</td>
        <td>0.888</td>
    </tr>
</table>

<table>
    <tr>
        <th rowspan=2>Dataset</th>
        <th colspan=3>v1</th>
        <th colspan=3>v2</th>
        <th colspan=3>v3</th>
        <th colspan=3>v4</th>
    </tr>
    <tr>
        <th>Training Time</th>
        <th>Test Time</th>
        <th>GPU Memory</th>
        <th>Training Time</th>
        <th>Test Time</th>
        <th>GPU Memory</th>
        <th>Training Time</th>
        <th>Test Time</th>
        <th>GPU Memory</th>
        <th>Training Time</th>
        <th>Test Time</th>
        <th>GPU Memory</th>
    </tr>
    <tr>
        <th>FB15k-237</th>
        <td>2 s / epoch</td>
        <td>< 1 s</td>
        <td>3.51 GiB</td>
        <td>6 s / epoch</td>
        <td>< 1 s</td>
        <td>5.02 GiB</td>
        <td>15 s / epoch</td>
        <td>< 1 s</td>
        <td>6.56 GiB</td>
        <td>29 s / epoch</td>
        <td>< 1 s</td>
        <td>8.10 GiB</td>
    </tr>
    <tr>
        <th>WN18RR</th>
        <td>3 s / epoch</td>
        <td>< 1 s</td>
        <td>5.17 GiB</td>
        <td>22 s / epoch</td>
        <td>< 1 s</td>
        <td>11.3 GiB</td>
        <td>3 s / epoch</td>
        <td>1 s</td>
        <td>18.7 GiB</td>
        <td>7 s / epoch</td>
        <td>1 s</td>
        <td>6.84 GiB</td>
    </tr>
</table>

## Citation ##

If you find this codebase useful in your research, please cite the original paper.

```bibtex
@article{zhu2021neural,
  title={Neural bellman-ford networks: A general graph neural network framework for link prediction},
  author={Zhu, Zhaocheng and Zhang, Zuobai and Xhonneux, Louis-Pascal and Tang, Jian},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
