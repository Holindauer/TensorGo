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