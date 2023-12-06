# Project Structure

This document describes the structure of the Go-LinAlg project. It is intended to be a guide for new contributors to the project.


## Directory Structure

- Tensor-Go is the root directory of the project. It contains various documentation for the project. 

- Tensor-Go/TG contains the source files for the library itself. This is where all of the functions are defined.

- Tensor-Go/Scripts contains various scripts for the project. 

- Tensor-Go/Tests contains the tests for the library. Each source file has a corresponding test file.

- Tensor-Go/Examples contains examples of how to use the library (currently under construction)


# Source File Structure

This next section describes the order you should familiarlize yourself with the source files in the TG directory.

## tensor.go

Within the TG directory, there are certain core files that impact the way the entire library is structured. The most important file of this nature, and the one you should look over first if you're interested in contributing to this project is [tensor.go](TG/tensor.go)

GLA is a library intended for working with arrays (Tensors) of arbitrary dimmension. [tensor.go](TG/tensor.go) is where the Tensor struct is defined. The Tensor struct is the core data structure of the library. It is a struct that contains a slice of floats, and a slice of ints that describe the multi-dimmensional shape of the tensor. 

It also contains functions for retrieving data within the tensor. These such functions are Index(), UnravelIndex(), Retrieve(), and Extract(). Having an understanding for what these functions do is important for understanding the theory behind how the library works.

## batching.go

You will notice that nearly every function in this library accepts a boolean argument called "batching". This is because operations on Tensors are designed to work on batches of Tensors (i.e. multiple Tensors at once). 

How does batching work? The [batching.go](TG/batching.go) files contains a variety of functions that generalize the process of performing an operation on a batch of Tensors of any number of elements. 

These operations are named after the their arguments and return value: Batch_Argument_ReturnType_Operation(). Batch_TwoTensor_Tensor_Operation() is an example of a function that takes two Tensors as input, and returns a single Tensor as output. 

batched operations take advantage of concurrency to perform operations on each element in the batch simulataneously. This is done using go routines from the "sync" package.


## shape.go

shape.go contains functions for manipulating the shape of a Tensor. These functions are part of the libraries core functionality. They are also re-used throughout the library. Having an understanding for what these functions do is important for understanding much of how the library works.

## init_tensor.go

init_tensor.go contains functions for initializing Tensors. 

## Tensor Operations ----> all_element_ops.go, all_inplace_ops.go, axis_collapsing_ops.go, axis_inplace_ops.go, vector.go, matrix.go, elimination.go

Above are source files that contain GLA's main operations that can be performed on a Tensor. The discintion between these files is in the specific type of operation they perform.

In the most general sense, a Tensor oepration either results in:
- A scalars, that is stored in a Tensor struct (This is so batching is possible without requiring polymorphism).
- or another Tensor.

These types of operations can then be further differentiated in the following manner. The following bulletin points describe these distinctions as well as the naming convention used for source files. 

all_ ops:

- Source files that begin with all_ perform operations that entail using all elements of a Tensor at once. This does not include elements across elements of a batch however. 
- If inplace_ follows all_, then the operation modifies a copy of the Tensor in place. 
- If element_ follows all_, then the operation results in a scalar output.

axis_ ops:
- Source files that begin with axis_ perform operations along a specific dimmension of a Tensor. 
- If inplace_ follows axis_, then the operation modifies a copy of the Tensor in place.
- If collapsing_ follows axis_, then the operation collapses the axis in which the operation is perforemd. This can be thought of as treating each element along the axis as its own tensor. The colapse of the axis is then the result of performing an operation that results in a scalar, thus 'collapsing' that axis into the scalar.

vector.go, matrix.go, elimination.go: 
- These source files contain operations that are more specific to a particular application. They still take adavntage of the operation types described above, but for the sake of organization, they are placed in their own files.

## extentensions.go
Some operations utilize remote repositories that are cloned into the Tensor-Go/Extensions directory. The extentions.go file directs scripts in the Tensor-Go/Scripts directory to clone these repositories. At current, this aspect of the library is still under construction and thus there is limited documentation on how to use it.
