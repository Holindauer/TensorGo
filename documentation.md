# Documentation
----------------------------------------------------------
The following is the documentation for this library.

# Tensors
This library introduces the Tensor data structure. All functions and methods in this library are used in relation to this data structure in some way. A Tensor is an n dimmensional array of numbers. Although Tensor structs can have any number of dimmensions, their data is stored in memory contiguously within their Data member. To simulate higher dimmensionality, the Shape member is used to calculate the 1D index with respect to a given multi-dimmensnional index. 

    A            // <--- Tensor struct
    A.Data       // <--- 1D array of float64 values
    A.BoolData   // <--- 1D array of bool values
    A.Shape      // <--- shape of the tensor
    
Data can be stored as either a float64 or a bool. However, most operations in this library are only supported for float64 Data.


# A Note About Batching...

This library supports batched Tensor operations. This means that a single Tensor struct can be used to represent a collection of individual Tensors. Every function in this library is capable of (or soon to be) operating on batched Tensors such that the operation is applied to the individual members of the batch, and not the Tensor as a whole. 

The way to specify if a function is to operate on a batched Tensor is to set it's last argument to True. The last argument of all functions in this library is a boolean value that specifies whether the operation is batched or not. 

    Sum_Tensor := A.Add(B, true)  // <--- A and B are batched Tensors

If a Tensor is batched, the first dimmension is used as the batch dimmension. 

    Batch_Rng := Range_Tensor([]int{3, 2, 2}, true)  // <--- 3 Tensors of shape (2, 2)


There is techincally no difference between the mechanism by which batched vs non-batched Tensors are stored in contiguous memory. Only in how their data is initialized and operated upon.

There is an optional Batch member that can be set to signify if a Tensor is intended for batched operations. This is only a flag and does not affect the Tensor in any way.

    A.Batched = true 

# Initialization of Tensors
A Tensor intializaton function will always require the Tensor shape and whether it should be batched. A Tensor initialization functions will always return a pointer to a Tensor struct. 

### Zero_Tensor(), Ones_Tensor(), Const_Tensor()
These functions are used to intialize Tensor structs which contain a constant float64 value across all elements. They all follow the same syntax

    var A *Tensor = FName(shape []int, batching bool)

### Range_Tensor()
Range Tensor has the same syntax as Const_Tensor(), however it will intialize its contiguous memory in a range starting from 0 and going to the number of elements in the Tensor. 

    var A *Tensor = Range_Tensor(shape []int, batching bool)

### RandFloat_Tensor()
RandFloat_Tensor() intitialized with values in a random range between specified min and max values. 

    var A *Tensor = RandFloat_Tensor(shape []int, batching bool, min float64, max float64)

### Copy() 
The Copy() method is used to copy an entire Tensor into a new Tensor. The new Tensor will have the same shape and batching as the original Tensor. 

    var A *Tensor = Zero_Tensor([]int{2, 2}, false)
    var A_Copy *Tensor = A.Copy()
    
### Eye() 
The Eye() function is used to initialize a 2D Identity matrix of a specified size. 

    var A *Tensor = Eye([]int{4, 3, 3}, false)  // <--- 4 3x3 Identity matricies

### Gram()
The Gram() fucntion creates a Matrix of an inputed matrix Tensor. This means that for a 2D Tensor A the Gram(A) will return A multplied by its Transpose. Batching is supported.

    var A *Tensor = RandFloat_Tensor([]int{2, 2}, 0, 1, false)
    var A_Gram *Tensor = Gram(A, false)  // <--- A_Gram = A * A.T


# Accessing Data From A Tensor

### Retrieve() 
The Retrieve() method is used to retrieve a single element from a Tensor. It takes a list of indices as input and returns the value at that index. 

    var A *Tensor = Zero_Tensor([]int{2, 2}, false)
    var A_00 float64 = A.Retrieve([]int{0, 0})

### Index()
Internally, the Retrieve() method calls the Index() method. Index() recieves the shape of a Tensor and a multi-dimmensional index, to which it returns the corresponding 1D index for that shape and indexing. A retrieved index can be used to directly access elements from the Data member of a Tensor.

    var A *Tensor = Ones_Tensor([]int{2, 2}, false)
    var flat_idx int = A.Index([]int{1, 1}, []int{2, 2})  // multi-dim idx, shape
    var A_11 float64 = A.Data[flat_idx] // retrieve value at index



### UnravelIndex()
UnravelIndex() is used to take a flat 1D index and transform it into a multidimmensional index given a Tensor shape. 

    var multi_dim_idx []int = UnravelIndex(flat_index, shape []int)  

### Extract()
Extract() is used to retrieve a Tensor element from a batched Tensor. It accepts an integer index of the first dimmension of the Tensor, representing the elemnt of the batch to extract. 

var element *Tensor = A.Extract(0)  // <--- Extract first element of batch


# Operations on Tensor Shape
There are various operations in this library used for manipulating the shape of a Tensor. 

### Partial()
The Partial method recieves a string containing a python style slice notation indexing of each dimmension of a Tensor. Partial() extracts the specified ranges of each dimmension of the Tensor it acts on into a new Tensor. The bounds of the slice are exclusive of the upper bound. By using just a colon for a dimmensions index, the entire dimmension is extracted.

    A := Range_Tensor([]int{3, 4, 9, 2})
    A_Partial := Partial(A, "0:2, 2:, :3, :")

### Reshape()
The Reshape() function accepts a new shape slice for a given Tensor. Assuming the new shape is of the same number of total elements, this shape will be swapped with the Tensor being acted on. The underlying contiguous memory will remain the same. 

    A := Range_Tensor([]int{3, 4, 9, 2})
    var A_reshaped *Tensor = A.Reshape([]int{2, 3, 6, 2})

### Transpose()
The Transpose() method recieves a permuation of a slice of integers ranging from 0 to len(dimmensions). Transpose() will reorder both the Shape member of the Tensor it acts on as well as the contiguous memory to reflect this change. 

    A := Range_Tensor([]int{3, 4, 9, 2})
    A_Reversed := A.Transpose([]int{3, 2, 1, 0})  
### Concat()
The Concat() method recieves arguments of an integer axis of concatenation and a Tensor pointer to which will be concatenated to the Tensor the method acts upon. Concat() requires that the Tensor Shapes be the same except for the axis of concatenation.

    A := Range_Tensor([]int{3, 4, 9, 2})
    B := Range_Tensor([]int{3, 4, 9, 2})
    var A_cat_B *Tensor = A.Concat(B, 0)  // <--- Concatenate B to A along first dimmension

### Extend_Shape()
The Extend_Shape() methods adds a new dimmension of a specified num_elements to the end of a Tensor. It returns a pointer to a new Tensor with these changes made. Values in the extended dimmension are intialized to zero.

    A := Range_Tensor([]int{3, 4, 9, 2})
    var A_extended *Tensor = A.Extend_Shape(5)  // shape: (3, 4, 9, 2, 5)

### Extend_Dim()
The Extend_Dim() appends additional elements to an already existing axis. New elements are initialized to zero. 

    A := Ones_Tensor(axis int, num_elements int)

### Remove_Dim()
The Remove_Dim() method recieves an integer axis_of_removal and element of retrieval. The axis of removal is removed from the Tensor, with the element_of_retrieval specifying which 'state' of that dimmension to retrieve out of it. This is a similar idea to specifying a single element out of a batch Tensor, then extracting only that element. 

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

    var A_Norm *Tensor := Norm(A *Tensor, batching bool)

### Unit() 
The Unit() function returns a unit vector in the direction of a given vector Tensor. 

    var A_Unit *Tensor = Unit(A *Tensor, batching bool)

### Check_Perpendicular()
The Check_Perpendicular() function returns a Zero_Tensor of either of shape [1] or of a [batch, 1] depending on the batching argument. The boolean of whether the Two vectors are perpindicular is stored within the boolData member of the Tensor.

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

# Operations Across All Elements
The following functions are Tensor operations applied to all elements of at once. 

### Sum_All(), Mean_All(), Var_All(), Std_All()
The functions compute the sum, mean, variance, or standard deviation across all elements of a Tensor at once. They each return a float64 value of the stat they compute. Currently there is no batching option. Their syntax is the same between them.

    var stat float64 := A.Sum_All()

### Add(), Subtract()

Add() and Subtract perform elementwise addition and subtraction respectively. Their syntax is the same between them.

    var A_plus_B *Tensor = A.Add(B, true)  

### Scalar_Mult() 
Scalar_Mult() accepts an float64 argument and performs scalar multiplication across a Tensor. Currently there is no batching option.

    var A_scaled *Tensor = A.Scalar_Mult(2.0) 

# Operations Across an Axis
The following funcitons perform an operation along a specified axis. The shape of the resulting Tensor will be 1 less than the original argument. This is because specified axis the operation is to be performed on is collapsed into itself. 

### Sum_Axis(), Mean_Axis(), Var_Axis(), Std_Axis()

The functions compute the sum, mean, variance, or standard deviation across a specified axis of a Tensor. They each return a pointer to a new Tensor with the result of the operation. Currently there is no batching option. Their syntax is the same between them.

    var A_summed *Tensor = A.Sum_Axis(0)  // <--- Sum along first dimmension
