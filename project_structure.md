# Directory Structure

This document describes the structure of the Tensor-Go project. It is intended to be a guide for new contributors to the project.


## Directory Structure

- Tensor-Go, the root directory, contains documentation for the project. 

- Tensor-Go/TensorGo contains the source code for the library.

- Tensor-Go/Tests contains tests of library functions

- Tensor-Go/Examples contains examples of ways to use the library 


# Source Code Structure

This section contains information on the different source files in Tensor-Go/TensorGo.


## Tensor.go
 
[Tensor.go](TensorGo/Tensor.go) is where the core data structure of this libary is defined, the Tensor. It is also where the different indexing functions for Tensors are found: Get(), Index(), TheoreicalIndex(), UnravelIndex(), and GetBatchElement().


## Types of Operations
Tensor-Go supports a few basic Tensor operation generalizations that are defined using some roughly accurate terminology from Abstract Algebra

An operation in TG is either *closed* or *open*

- A closed operation takes an input and returns an outpout in the same space/set. IE: A Tensor is input to an operation and the output is a Tensor of the same shape/space.

- Whereas an open operation is not restricted by input and output shapes. Open operation arguments can be scalars, differently shaped tensors, etc... For example, A Tensor input returns a scalar output.

Closed and Open Operations are then further subdivided into their argument types:

- *Closed* Unary, Binary, Ternary... operations in Tensor-Go accept 1,2,3... Tensor arguments of the same dimmension and return a Tensor with the same dimmension as the inputs. For example, elementwise Tensor addition is a closed binary operator because it requires two inputs of the same shape and outputs a single Tensor with that same shape.

- *Open* Unary, Binary, Ternary... operations in Tensor-Go accept 1,2,3... Tensor arguments (not necessarily of the same set/space) and output a Tensor that is not in the original set/space. For example, taking the mean along a Tensor's axis is an open unary Tensor operation because it outputs a Tensor with one less dimmension than the original. Another example is broadcasted addition of a matrix with a vector, which is an open binary Tensor operation, because it accepts two Tensor inputs of different dimmensions and brings it to a single dimmension

The key distinction between Open and Closed operations is that in closed operations, the elements of the input and output share the same Tensor space between them all. Whereas this is not the case for open ops.

These functions are found in:  
- [ClosedBinaryOps.go](TensorGo/ClosedBinaryOps.go)
- [ClosedUnaryOps.go](TensorGo/ClosedUnaryOps.go)
- [OpenBinaryOps.go](TensorGo/OpenBinaryOps.go)
- [OpenUnaryOps.go](TensorGo/OpenUnaryOps.go)


## BatchedOp.go

You will notice that nearly every function in this library accepts a boolean argument called "batching". This is because operations on Tensors are designed to work on batches of Tensors (i.e. multiple Tensors at once). 

[BatchedOps.go](TensorGo/BatchedOps.go) contains interfaces for generalizing the process of performing an operation aross a batch of Tensors. Nearly every function in this Tensor-Go accepts a boolean argument called "batching". If set to true, the function will be performed using one of the functions in BatchedOps.go to distributed the operation across batch elements.

## AutoGrad.go and NeuralNetwork.go
 [AutoGrad.go ]( TensorGo/AutoGrad.go ) contains an implementation of reverse mode automatic differentiation for the use of backpropogation in neural network training. This is a special functionality that involves maintaing a directed acyclic graph of all computation involved in creating a specific scalar value. This computational graph also tracks the gradient computation at each node for backpropogation. 

 [NeuralNetwork.go](TensorGo/NeuralNetwork.go) allows for the creation of neural networks (atm only mlps). Neural nets use AutoGrad.go to track the gradient of the loss wrt each parameter in the network.


## ShapeOps.go

[ShapeOps.go contains](TensorGo/ShapeOps.go) functions for manipulating the shape of a Tensor. These functions are part of the libraries core functionality. They are also re-used throughout the library. Having an understanding for what these functions do is important for understanding much of how the library works.

## Linear Algebra Functionality

Linear Algebra functionality can be found in:

- [VectorOps.go](TensorGo/VectorOps.go) 
- [MatrixOps.go](TensorGo/MatrixOps.go) 
- [LinearSystemsOps.go](TensorGo/LinearSystemsOps.go)



## IntiTensor.go

[IntiTensor.go](IntiTensor.go) contains functions for initializing Tensors with certain properties. 

## Save.go 

[Save.go](TensorGo/Save.go) contains functions for saving/loading Tensors.