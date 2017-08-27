# meProp
Codes for "meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting"

The MNIST part.

## Requirements
### C#
* Targeting Microsoft .NET Framework 4.6.1
* Compatible versions of Mono should work fine (tested on Mono 5.0.1)
* Developed with Microsoft Visual Studio 2017
### PyTorch
* Python 3.5
* Pytorch v0.1.12 (we do not know whether the code is compatible with v0.2.0 yet)
* CUDA 8.0 if using GPUs
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
where <config.json> is a configuration file. There is an example configuration file in the source codes.
### PyTorch
```bash
python3.5 meprop (PyTorch).py
```
## Cite
To use the codes or compiled executables, please cite the following paper:

Xu Sun, Xuancheng Ren, Shuming Ma, Houfeng Wang. 
meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting. In proceedings of ICML 2017.
[[arxiv]](https://arxiv.org/abs/1706.06197)