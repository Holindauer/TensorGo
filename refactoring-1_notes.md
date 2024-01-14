# refactoring-1 Branch of Tensor-Go 

In this branch, I'm going to do some major refactoring of the code base. The following goals are a priority:

- Switch to the NatSpec commending standard. After writing a bit of solidity, I've come to appreciate this commenting standard. It's a bit more verbose, but it's much more readable.
- Rebranding/renaming of source files. Instead of using the all_axis_ops type of naming that I have been using, I am going to switch to a more standardized naming convention based on abstract algebra. For example, binary_ops, unary_ops, reduction_ops, etc.
- Refactoring of major functions to reduce code duplication. 


# Changes Made:

- Retrieve() has been changed to Get()
- Index_Off_Shape() is now TheoreticalIndex()
- Transpose() --> Permute()
- Partial() --> Slice()


# Distinction Different Types of Tensor Operations

The different types of operations described/Implemented in Tensor-Go reflect the terminology of Abstract Algebra

An operation in TG is either *closed* or *open*

- A closed operation takes an input and returns an outpout in the same space/set. IE: A Tensor is input to an operation and the output is a Tensor of the same shape/space.

- Whereas an open operation accepts an input in one space/set and brings it to another. For example, A Tensor input returns a scalar output.

Closed and Open Operations are then further subdivided into their argument types:

- *Closed* Unary, Binary, Ternary... operations in Tensor-Go accept 1,2,3... Tensor arguments of the same dimmension and return a Tensor with the same dimmension as the inputs. For example, elementwise Tensor addition under a closed binary operator.

- *Open* Unary, Binary, Ternary... operations in Tensor-Go accept 1,2,3... Tensor arguments (not necessarily of the same set/space) and output a Tensor that is not in the original set/space. For example, taking the mean along a Tensor's axis is an open unary Tensor operation because it outputs a Tensor with one less dimmension than the original. Another example is broadcasted addition of a matrix with a vector, which is an open binary Tensor operation, because it accepts two Tensor inputs of different dimmensions and brings it to a single dimmension

The key distinction between Open and Closed operations is that in closed operations, the elements of the input and output share the same Tensor space between them all. Whereas this is not the case for open ops.