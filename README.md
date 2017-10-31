# meProp

The codes were used for experiments on MNIST with _meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting_ (ICML 2017) [[pdf]](http://proceedings.mlr.press/v70/sun17c/sun17c.pdf) by Xu Sun, Xuancheng Ren, Shuming Ma, Houfeng Wang. Both the codes for CPU and GPU were included.

We propose a simple yet effective technique for neural network learning, which we call **meProp** (*m*inimal *e*ffort back *prop*agation). The forward propagation is computed as usual. In back propagation, only a small subset of the full gradient is computed to update the model parameters. The gradient vectors are sparsified in such a way that **only the top-k elements (in terms of magnitude) are kept**. As a result, only *k* rows or columns (depending on the layout) of the weight matrix are modified, leading to a linear reduction (*k* divided by the vector dimension) in the computational cost. Surprisingly, experimental results demonstrate that we can **update only 1–4% of the weights** at each back propagation pass. This does not result in a larger number of training iterations. More interestingly, the proposed method **improves the accuracy of the resulting models** rather than degrades the accuracy, and a detailed analysis is given in the paper.

![An illustration of the idea of meProp.](./docs/illustration.svg)

**TL;DR**: Training with meProp is significantly faster than the original back propagation, and has better accuracy on all of the three tasks we used, POS Tagging, Dependency Parsing, and MNIST respectively. The method works with different neural models (MLP and LSTM), with different optimizers (we tested AdaGrad and Adam), with DropOut, and with more hidden layers. The top-*k* selection works better than the random k-selection, and better than normally-trained *k*-dimensional network.

Results on test set (please refer to the paper for detailed results and experimental settings):

| Method (AdaGrad, CPU)   | Backprop Time (s) | Test (%)          |
| ----------------------- | ----------------- | ----------------- |
| POS-Tag (LSTM 500d)     | 17,534            | 96.93             |
| POS-Tag (meProp top-5)  | **253 (69.2x)**   | **97.25 (+0.32)** |
| Parsing (MLP 500d)      | 8,900             | 88.92             |
| Parsing (meProp top-20) | **492 (18.1x)**   | **88.95 (+0.03)** |
| MNIST (MLP 500d)        | 171               | 97.52             |
| MNIST (meProp top-10)   | **4 (41.7x)**     | **98.00 (+0.48)** |

The effect of *k*, selection (top-*k* vs. random), and network dimension (top-*k* vs. *k*-dimensional):

![Effect of k](./docs/effect-k.png)

To achieve speedups on GPUs, a slight change is made to unify the top-_k_ pattern across the mini-batch. The original meProp will cause different top-_k_ patterns across examples of a mini-batch, which will require sparse matrix multiplication. However, sparse matrix multiplication is not very efficient on GPUs compared to dense matrix multiplication on GPUs. Hence, by unifying the top-_k_ pattern, we can extract the parts of the matrices that need computation (dense matrices), get the results, and reconstruct them to the appropriate size for further computation. This leads to actual speedups on GPUs, although we believe if a better method is designed, the speedups on GPUs can be better.

See [[pdf]](https://arxiv.org/abs/1706.06197) for more details, experimental results, and analysis.

bibtex:
```
@InProceedings{sun17meprop,
  title = 	 {me{P}rop: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting},
  author = 	 {Xu Sun and Xuancheng Ren and Shuming Ma and Houfeng Wang},
  booktitle = 	 {Proceedings of the 34th International Conference on Machine Learning},
  pages = 	 {3299--3308},
  year = 	 {2017},
  volume = 	 {70},
  series = 	 {Proceedings of Machine Learning Research}
}
```

# Usage

We developed the codes for the purpose of research, and we do not guarantee the codes are fully annotated and easy to understand, although we did annotate the key parts and adjust the structure before publishing the codes. 

## Requirements

### C#
* Targeting Microsoft .NET Framework 4.6.1
* Compatible versions of Mono should work fine (tested Mono 5.0.1)
* Developed with Microsoft Visual Studio 2017
### PyTorch
* Python 3.5
* PyTorch v0.1.12 (we do not know whether the code is compatible with v0.2.0 yet)
* torchvision
* CUDA 8.0


## Dataset

### C#
MNIST: Download from [link](http://yann.lecun.com/exdb/mnist/). Extract the files, and place them at the same location with the executable.

### PyTorch
MNIST: The code will automatically download the dataset and process the dataset (using torchvision). See function _get_mnist_ in the pytorch code for more information.

## Run
### C#
Compile the code first, or use the executable provided in releases.

Then
```
nnmnist.exe <config.json>
```
or
```
mono nnmnist.exe <config.json>
```
where <config.json> is a configuration file. There is [an example configuration file]("./meprop (CSharp)/nnmnist/default.json") in the source codes. The output will be written to a file at the same location with the executable. The code supports random top-_k_ selection in addition.
### PyTorch
```bash
python3.5 meprop (PyTorch).py
```
The results will be written to stdout by default, but you can change the argument _file_ when initializing the _TestGroup_ to write the results to a file. The code supports simple unified meProp in addition. Please notice, this code will use GPU by default.