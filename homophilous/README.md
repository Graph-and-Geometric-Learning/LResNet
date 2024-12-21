## LResNet in GNN
This is the implementation of LResNet as residual connection in GNN architectures, where the base model is HyboNet from [*Fully Hyperbolic Neural Networks*](https://arxiv.org/abs/2105.14686).

### Usage
We provide several examples of running our implementation on the datasets provided in the **data** folder. To run our code, first run 
```
bash set_env.sh
```
For link prediction, run
```
bash example/example_lp.sh
```
For node classification, run
```
bash example/example_nc.sh
```
The results will be saved to an **output** file. 