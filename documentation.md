# Documentation
----------------------------------------------------------
The following is the documentation for this library.

# Tensors
Tensor-Go uses the Tensor data structure. A Tensor is an array of arbitrary dimmesionality. Although Tensors can have any dimmensionality, under the hood Tensor-Go stores Tensor data contiguously. For a Tensor A, contiguous memory is stored in the *Data* member of the Tensor struct, and the *Shape* member is used to keep track of the dimmensions of the Tensor.

    A            // <--- Tensor struct
    A.Data       // <--- 1D array of float64 values
    A.BoolData   // <--- 1D array of bool values
    A.Shape      // <--- shape of Tensor A
    A.Batched    // <--- flag for batched operations
    
Currently, Tensor-Go supports float64 and bool Tensor data. However, most operations in this library are currently only supported for float64 Data.

# Indexing Tensors

As mentioned above, the multidimmensionality of Tensors in Tensor-Go is simulated by the use of the *Shape* member. This is an integer slice that contains the number of elements along each axes of a multidimmensional Tensor. 

Tensor-Go takes handles the conversion of multi dimmensional indicies back to their corresponding 1D index in memory. The following are various ways to access elements of a Tensor.

### Retrieve() 
The Retrieve() method is used to retrieve a single element from a Tensor given an integer slice representing the multi-dimmensional index of the element to retrieve. Currently, Retrieve is only supported for float64 data.

    var A float64 = (A *Tensor) Retrieve(indices []int)

### Index()
Internally, the Retrieve() method calls the Index() method. Index() recieves the shape of a Tensor and a multi-dimmensional index, to which it returns the corresponding 1D index for that shape and indexing. A retrieved index can be used to directly access elements from the Data member of a Tensor.

    var flat_index int = (A *Tensor) Index(indices []int) 

### Index_Off_Shape()
Additionally, Index_Off_Shape() is used to retrieve the 1D index of a Tensor given a multi-dimmensional index and a shape without the need of a Tensor struct.

    var flat_index int = Index_Off_Shape(indices []int, shape []int)


### UnravelIndex()
UnravelIndex() is used to convert a flat 1D index into a multidimmensional index given a Tensor shape. 

    var multi_dim_idx []int = UnravelIndex(flat_index, shape []int)  

### Extract()
Extract() is used to retrieve a Tensor element from a batched Tensor. It accepts the integer index of the first dimmension of the Tensor containing the element of the batch to extract. 

    var batch_element *Tensor = (A *Tensor) Extract(batch_element int) 


# A Note About Batching...

Tensor-Go supports *batched Tensor operations*. 

This means that a single Tensor can be used to hold many other individual Tensors. The elements of a batched Tensor are stored in the 0'th dimmension of the Tensor. *Technically, this is only a symbolic representation of the data. The underlying contiguous memory of a batched Tensor is the same as a non-batched Tensor. Because of this, an optional Batched flag can be set in a Tensor struct to clarify it is a batch.*

    A.Batched = true


Every function in this library is capable of operating in a batched fashion. The last argument of every function in this library is a boolean value that specifies whether the operation is batched or not. 

    Sum_Tensor := A.Add(B, true)  // <--- A and B are batched Tensors

If a Tensor is batched, the first dimmension is used as the batch dimmension. The operation will be applied to every element along that dimmension.

# A note about Broadcasting

Tesnor-Go support *broadcasting*.

 Broadcasting is the process of performind a Tensor operation across two Tensors of different shapes. The smaller Tensor is 'broadcasted' across the larger Tensor to match its shape.

For Example, if you have a [3, 3, 3] batched Tensor and you want to add a [3, 3] Tensor to each element of the batch, you can broadcast the [3, 3] Tensor across the 0'th axis of the [3, 3, 3] to match the larger shape. 



### Broadcast()
Broadcast() allows you to turn any already existing operation into a Broadcasted operation. The Broadcast_Arg is the Tensor we are broadcasting onto Broadcast_Onto. The op argument is a function that accepts two Tensor pointers and returns a Tensor pointer. When defining the op function, the first argument is the Broadcast_Arg and the second argument is the Broadcast_Onto.

    var broadedcasted *Tensor = (Broadcast_Arg *Tensor) Broadcast(Broadcast_Onto *Tensor, op func(A *Tensor, B *Tensor) *Tensor) 

There are also some pre-defined broadcasted operations listed below which are explained along in the documentaion along with their non-broadcasted counterparts. 
- Broadcast_Add() 
- Broadcast_Subtract()

# Initialization of Tensors
Tensor-Go provides various functions for intializing Tensors with different attributes.

The general format for these type of functions is as follows (with some variation).

    var A *Tensor = FName(shape []int, batching bool)

### Zero_Tensor(), Ones_Tensor(), Const_Tensor()
The above functions are used to intialize Tensor structs that contain a constant float64 value across all elements. They all follow the same syntax

    var zero *Tensor = Zero_Tensor(shape []int, batching bool)
    var ones *Tensor = Ones_Tensor(shape []int, batching bool)
    var constant *Tensor = Const_Tensor(shape []int, value float64, batching bool)

### Range_Tensor()
Range_Tensor() intializes a Tensor with its contiguous memory in a range starting from 0 to the total number of elements in the Tensor. 

    var range *Tensor = Range_Tensor(shape []int, batching bool)

### RandFloat64_Tensor()
RandFloat_Tensor() intitialized with float64 values in a random range between specified min and max values. 

    var random *Tensor = RandFloat64_Tensor(shape []int, lower float64, upper float64, batching bool) *Tensor 
### Copy() 
The Copy() method is used to create a deep copy of a Tensor. There is currently no option for batching.

    var A *Tensor = Zero_Tensor([]int{2, 2}, false)
    var deep_copy_A *Tensor = A.Copy()
    
### Eye() 
The Eye() function is used to initialize a 2D Identity matrix of a specified size. 

    var I *Tensor = Eye(shape []int, batching bool) 

### Gram()
The Gram() fucntion computes the Matrix product of a Tensor with its transpose. It is used to compute the Gram Matrix of a Tensor.

    var gram *Tensor = A.Gram(batching bool) 


# Operations on Tensor Shape

Manipulating the shape of Tensors is a useful operation when working wiht multi-dimmensional data. The following functions provide various ways to manipulate the shape of a Tensor.

*Note: Currently batching is limited for shape related functions.*

### Partial()
The Partial method recieves a string containing a python style slice notation indexing of each dimmension of a Tensor. Partial() extracts the specified ranges of each dimmension of the Tensor it acts on into a new Tensor. The bounds of the slice are exclusive of the upper bound. By using just a colon for a dimmensions index, the entire dimmension is extracted.

    // Syntax:
    var partial *Tensor := Partial(slice string)

    // For Example:
    A := Range_Tensor([]int{3, 4, 9, 2})
    A_Partial := Partial(A, "0:2, 2:, :3, :")

### Reshape()
The Reshape() function accepts a new shape slice for a given Tensor. Assuming the new shape is of the same number of total elements, this shape will be swapped with the Tensor being acted on. The underlying contiguous memory will remain the same. 

    var reshaped *Tensor = A.Reshape(new_shape []int, batching bool)

*Note: For batched reshapes, only specify the new shape of the batch elements.*

### Transpose()
The Transpose() method recieves a permuation of a slice of integers ranging from 0 to len(dimmensions). Transpose() will reorder both the Shape member of the Tensor it acts on as well as the contiguous memory to reflect this change. 

    // Syntax:
    var transposed *Tensor := Transpose(permuation []int)

    // Example:
    A := Range_Tensor([]int{3, 4, 9, 2})
    A_Reversed := A.Transpose([]int{3, 2, 1, 0})  

### Concat()
The Concat() method accepts an integer axis of concatenation and a Tensor pointer to which will be concatenated to the Tensor the method acts upon. Concat() requires that the Tensor Shapes be the same except for the axis of concatenation.

    // Syntax:
    var concatenated *Tensor := A.Concat(B *Tensor, axis_cat int)

    // Example:
    A := Range_Tensor([]int{3, 4, 9, 2})
    B := Range_Tensor([]int{3, 4, 9, 2})
    var A_cat_B *Tensor = A.Concat(B, 0)  // <--- Concatenate B to A along first dimmension

### Extend_Shape()
The Extend_Shape() methods adds a new dimmension of a specified num_elements to the end of a Tensor. It returns a pointer to a new Tensor with these changes made. Values in the extended dimmension the same as the original Tensor across all elements in the new axis.

    var A_extended *Tensor = A.Extend_Shape(num_elements int)

### Extend_Dim()
The Extend_Dim() appends additional elements to an already existing axis. New elements are initialized to zero. 

    var A_extended *Tensor = A.Extend_Dim(axis int, num_elements int)

### Remove_Dim()
The Remove_Dim() method recieves an integer axis_of_removal and element of retrieval. The axis of removal is removed from the Tensor, with the element_of_retrieval specifying which 'state' of that dimmension to retrieve out of it. This is a similar idea to specifying a single element out of a batch Tensor, then extracting only that element. 

    // Syntax:
    var removed *Tensor := Remove_Dim(axis_of_removal int, element_of_retrieval int)

    // Example:
    var Removed_0 *Tensor = A.Remove_Dim(0, 0)  // <--- Remove first dimmension, retrieve first element

### Remove_Singletons()
Remove_Singletons() removes all dimmensions of the Shape member that are of length 1. This does not change the underlying contiguous memory. In fact, a singleton dimmension in general does not affect operations on Tensors. 

    A_Squeezed := A.Remove_Singletons()

### Add_Singleton()
Add_Singleton() appends a 1 to the end of the shape of a Tensor. It does not affect the underlying contiguous memory. 

    A := Range_Tensor([]int{3, 4, 9, 2})
    var A_Singleton *Tensor = A.Add_Singleton()

There is another way to accomplish this as well that is more flexible to adding singletons at dimmensions other that the last. That is to directly manipulate the shape of the Tensor in question. 

    A.Shape = append(A.Shape, 1)  // <--- Add singleton dimmension to end of shape

    A.Shape = append(1, A.Shape...)  // <--- Add singleton dimmension to beginning of shape


# Vector Operations 
The following are operations for vector Tensors (ie: Tensors of a single dimmension). 

### Dot()
The Dot() function accepts two vector Tensors and returns their dot product in the form of a Tensor of 1 element or a batched Tensor of 1 element Tensors depending on the batching argument.

    var A_Dot_B *Tensor := Dot(A *Tensor, B *Tensor, batching bool)

### Norm()
The Norm() function returns the norm of a vector Tensor in the form of a Tensor of 1 element or a batched Tensor of 1 element Tensors depending on the batching argument.

    var A_Norm *Tensor := (A *Tensor) Norm(batching bool)

### Unit() 
The Unit() function returns a unit vector in the direction of a given vector Tensor. 

    var A_Unit *Tensor = (A *Tensor) Unit(batching bool)

### Check_Orthogonal(), Check_Acute(), Check_Obtuse()
The above function returns a Zero_Tensor of either of shape [1] or of a [batch, 1] depending on the batching argument. The boolean of whether the Two vectors are what the function is checking for. This data is stored within is stored within the BoolData member of the Tensor.

    var A_perp_B *Tensor = Check_Perpendicular(A *Tensor, B *Tensor, batching bool)

### Cosine_Similarity()
The Cosine_Similarity() function returns the cosine similarity of two vector Tensors. The scalar similarity score is returned in the form of a Tensor of 1 element or a batched Tensor of 1 element Tensors depending on the batching argument.

    var Similarity_Score *Tensor := Cosine_Similarity(A *Tensor, B *Tensor, batching bool)


### Outer() 
The Outer() function computers the outer product of a Tensor. It is essentially hte same function as MatMul(), but with added protections against improper shape. There is optional batching.


    var A_outer_B *Tensor = Outer(A *Tensor, B *Tensor, batching bool)


# Matrix Operations

### MatMul()
MatMul() performs the matrix multiplication of a Tensor A and B, assuming they are both 2D. 

    MatProd := MatMul(A, B, true) // <-- batched matmul

### Display_Matrix()
Display_Matrix() prints out a 2D Tensor. If batching is set to true, each 2D element of a Tensor will be printed in consecutive order. 

    Display_Matrix(A, true)


### Augment_Matrix()
Augment_Matrix() is primarly used within the Gaussian_Elimination() and Gauss_Jordan_Elimination() functions. However, it also can be called on its own. There is currently no batching option.

    var Aug_AB *Tensor := Augment_Matrix(A, B) 

# Linear Systems Solvers
The following are funcitons used to solve for x in Ax = b.

### Gaussian_Elimination()
    x := Gaussian_Elimination(A, b, true)  // <--- batched Gaussian Elimination

### Gauss_Jordan_Elimination()
    x := Gauss_Jordan_Elimination(A, b, true)  // <--- batched Gauss Jordan Elimination

### LinSys_Approximator()
The LinSys_Approximator() is an experimental feature that will accept A and b Tensors of a linear system, along with matrixType ("dense" or "sparse") and fillPecentage (0.0 to 1.0) arguments. The function will direct a process that trains a neural network on the spot for approximating the solution to linear systems as specified. The function will then run inference on that network to return the solution to the linear system.

    var x *Tensor = LinSys_Approximator(A *Tensor, b *Tensor, matrixType string, fillPercentage float64, batching bool) 

This is currently an experimental feature. It is not recommended to use this function for any serious work.

# Operations Across All Elements
The following functions are Tensor operations applied to all elements of at once. 

### Sum_All(), Mean_All(), Var_All(), Std_All()
The functions compute the sum, mean, variance, or standard deviation across all elements of a Tensor at once. They each return a float64 value of the stat they compute. Currently there is no batching option. Their syntax is the same between them.

    var stat float64 := A.Sum_All()

### Add(), Subtract()

Add() and Subtract perform elementwise addition and subtraction respectively. Their syntax is the same between them.

    var A_plus_B *Tensor = A.Add(B, true) 

There are also broadcasted versions of these functions.

    var A_broadcastedTo_B *Tensor = A.Broadcast_Add(B)
    var A_broadcastedTo_B *Tensor = A.Broadcast_Subtract(B)


### Scalar_Mult() 
Scalar_Mult() accepts an float64 argument and performs scalar multiplication across a Tensor. Currently there is no batching option.

    var A_scaled *Tensor = A.Scalar_Mult(2.0) 

# Operations Across an Axis
The following funcitons perform an operation along a specified axis. The shape of the resulting Tensor will be 1 less than the original argument. This is because specified axis the operation is to be performed on is collapsed into itself. 

### Sum_Axis(), Mean_Axis(), Var_Axis(), Std_Axis()

The functions compute the sum, mean, variance, or standard deviation across a specified axis of a Tensor. They each return a pointer to a new Tensor with the result of the operation. Currently there is no batching option. Their syntax is the same between them.

    var A_summed *Tensor = A.Sum_Axis(0)  // <--- Sum along first dimmension



