# src

This directory contains source codes for the papers.

## csharp

Directory csharp contains the code of meProp and meSimp, written in C#. This code runs on CPUs.

We have reworked the code. The code can train MLP models for MNIST using meProp, or meSimp. All that needs to be done is to modify the configuration file. 

We coded a simple framework for neural networks. It is similar to the dynamic computation graph method. The operations in the forward propagation is recorded, and in back propagation the corresponding gradient operations are applied.



The structure of the codes and some comments.

```
csharp\
-- nnmnist\
       Config.cs: configurations for the training
       default.json: an example of the configuration files
---- Application\
       the main training loop is in Mnist.cs
---- Common\
       codes for utilities: logger, RNG, timer, topn
---- Data\
       codes for loading the MNIST data
---- Networks\ 
       the codes for the neural network models
       MLP.cs: baseline
       MLPTop.cs: meProp
       MLPVar.cs: meSimp
       MLPRand.cs: random k selection of meProp
------ Graph\
         the basics for building neural networks
         Matrix.cs: a two-dimensional array for efficiency
         Tensor.cs: has weight, gradient, and the gradient history
         Flow.cs: the operations, and the record of the applied operations    
------ Inits\
         weight initialization methods
------ Opts\
         optimizers
------ Units\
         modules, e.g., the fully-connected layer
       
```


- To see the general training procedure, please refer to Mnist.cs.
- To see the implementation of meProp:
  - TopNHeap.cs contains the code for extracting the top-k elements.
  - StepTopK methods of the classes in the Units show the forward propagation procedure.
  - MultiplyTop methods in Flow.cs show the computation of the forward propagation and backward .propagation involving sparse multiplication of two matrices.
  - MLPTop.cs defines the model of meProp.
- To see the implementation of meSimp:
  - Mnist.cs contains the cycle mechanism.
  - DenseMaskedUnit.cs shows an equivalent way to remove the neurons by masking.
  - Record.cs keeps the activeness of the neurons.
  - MultiplyTopRecord methods in Flow.cs shows how the activeness is collected.
  - MLPVar.cs defines the model of meSimp.
- To see the neural network framework, please refer to the Graph directory.
  - Flow.cs defines the operations with the forward computation, and the backward gradient computation.
  - Tensor.cs defines the object of the operations in Flow.cs. Tensor has its value, its gradient, and the gradient history if required by the optimizer, e.g., Adam.


## pytorch

Directory pytorch contains the code of meProp, written in python, using the PyTorch package. The code runs on NVIDIA GPUs by default. It should be easy to make it run on CPUs.

We have also refactored the code. The code is for training MLPs for MNIST using meProp. In addition, unified meProp is supported. The configuration is set in the get_args function or the get_args_unified function. The configuration can also be set by passing arguments when running main.py


The structure of the codes and some comments.

```
pytorch\
  data.py 
    load MNIST data
  functions.py
    define forward and backward procedures for meProp (equivalent to torch.nn.functional.Linear)
  main.py
    run the experiments
  model.py
    define the MLP model for MNIST
  modules.py
    define modules for meProp (equivalent to torch.nn.Linear)
       
```


- To see the general training procedure, please refer to util.py.
- To see the implementation of meProp and unified meProp, please refer to functions.py.
- To see the MLP structure used, please refer to modules.py and model.py.




