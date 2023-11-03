# Go-LinAlg
GoLang Linear Algebra Library


The goal of this repository is to create an linear algebra library in golang that
takes advantage of concurrency. The ideas and formulas used within this repository will 
come from Gilbert Strang's Introduction to Linear Algebra.

# Documentation:
The following is an explanation of how Tensors are represened in this library, and how to use and perform operations on them.

-----------------------------------------------------------------------------------------------------

# Tensors
The basic datatype of this library are tensors, ie multidimensional arrays. In memory 
tensors are represented as a 1D slice of floats. However, this library provides methods
for accessing and manipulating the data as if it were a multidimensional array. Below 
are some functions and methods that can be used to create and access data from Tensors.

### Zero_Tensor(), Ones_Tensor(), Const_Tensor()
The above funcitons are used to create a tensor of a given shape with a uniform value
at each element. Const_Tensor() accepts a float64 as a second parameter that represents
the value of each element. The functions return a pointer to a tensor. Zero_Tensor()
and Ones_Tensor() internally call Const_Tensor() with the vals their names suggest, but
only require the shape of the tensor as a parameter.

    var tensor *Tensor = Zero_Tensor([]int{2,3,4})
    var tensor *Tensor = Ones_Tensor([]int{2,3,4})
    s
    var tensor *Tensor = Const_Tensor([]int{2,3,4}, 5)

### Range_Tensor() 
The Range_Tensor() function accepts a slice of integers that represent the dimensions
of the tensor. The function returns a pointer to a tensor initialize with the range of

    var tensor *Tensor = Range_Tensor([]int{2,3,4})

### tensor.Retrieve()
The Retrieve() function accepts a slice of integers that represent the index of the
element you want to access. The function returns the value of the element at the given
index. The function will panic if the index is out of bounds.

    var tensor *Tensor = New_Tensor([]int{2,3,4})
    var element float64 = tensor.Retrieve([]int{1,2,3})

### Index()
A tensor can also be indexed using the Index() function. Tensor data is represented in 
memory as a 1D contiguous slice of floats, regardless of the dimensionality of the
tensor. The benefit of this method of storage is that there is no need for a Flatten() 
function, since the data is flattened by default. This function is called internally 
within the tensor.Retrieve() method.

The Index() function accepts a slice of integers that represent the index
of the element you want to access and a slice representing the dimensions of the
tensor. The function returns the index of the element in the 1D slice.

    var tensor *Tensor = New_Tensor([]int{2,3,4})
    var index int = Index([]int{1,2,3}, []int{2,3,4})
    
    var element float64 = tensor.data[index]

### Same_Shape()
The Same_Shape() function accepts two Tensor pointers and returns true if the two
tensors have the same shape. 

    var tensor1 *Tensor = New_Tensor([]int{2,3,4})
    var tensor2 *Tensor = New_Tensor([]int{2,3,4})
    
    var same_shape bool = Same_Shape(tensor1, tensor2)

### Copy() 
The Copy() function accepts a Tensor pointer and returns a pointer to a new Tensors
that is a copy of the original tensor.

    var tensor1 *Tensor = New_Tensor([]int{2,3,4})
    var tensor1_copy *Tensor = Copy(tensor1)

### Eye()
The Eye() function accepts an integer and returns a pointer to a tensor that is a
2D identity matrix of the given size.

    var tensor *Tensor = Eye(5)

### Partial() 
The Partial() function accepts a Tensor pointer and a string representing that contains a 
python style slice. The function returns a pointer to a new tensor that is a partial
copy of the original tensor. The function will panic if the slice is not valid.

    var tensor *Tensor = New_Tensor([]int{2,3,4})
    var tensor_partial *Tensor = Partial(tensor, "0:1,1:3,2:4")

-----------------------------------------------------------------------------------------------------

# Vector Operations
The following operations are defined for vectors, ie 1D tensors.

### Dot Product
The Dot() function accepts two tensors and returns the dot product of the two tensors.
The function will panic if the dimensions of the two tensors are not compatible.

    var tensor1 *Tensor = New_Tensor([]int{5})
    var tensor2 *Tensor = New_Tensor([]int{5})
    
    tensor1.data = {1,2,3,4,5}
    tensor2.data = {1,2,3,4,5}

    var length_squared float64 = Dot(tensor1, tensor2)

### Outer Product
The Outer_Product() function accepts two tensors and returns the outer product of the
two tensors. The function will panic if the dimensions of the two tensors are not
both 1D.

    var tensor1 *Tensor = New_Tensor([]int{5})
    var tensor2 *Tensor = New_Tensor([]int{12})
    
    tensor1.data = {1,2,3,4,5}
    tensor2.data = {1,2,3,4,5,6,7,8,9,10,11,12}

    var outer_product *Tensor = Outer_Product(tensor1, tensor2) // shape: {5,12}

### Norm() 
The Norm() function accepts a tensor and returns the norm of the tensor.

    var tensor *Tensor = New_Tensor([]int{5})
    tensor.data = {1,2,3,4,5}

    var norm float64 = Norm(tensor)         
    length = math.Sqrt(Dot(tensor, tensor))  // <--- equivalent to Norm()

### Unit() 
The Unit() function accepts a tensor and returns a unit vector in the same direction.

    var tensor *Tensor = Range_Tensor([]int{5})
    tensor.data = {1,2,3,4,5}

    var unit_vector *Tensor = Unit(tensor)

### Check_Perpendicular()
The Check_Perpendicular() function accepts two tensors and returns true if the two
tensors are perpendicular (if their dot product is zero). The function will panic if
the dimensions of the two tensors are not compatible.

    var tensor1 *Tensor = New_Tensor([]int{5})
    var tensor2 *Tensor = New_Tensor([]int{5})
    
    tensor1.data = {1,2,3,4,5}
    tensor2.data = {1,2,3,4,5}

    var perpendicular bool = Check_Perpendicular(tensor1, tensor2)

### Cosine_Similarity()
The Cosine_Similarity() function accepts two tensors and returns the cosine similarity
of the two tensors. The function will panic if the dimensions of the two tensors are
not compatible.

    var tensor1 *Tensor = New_Tensor([]int{5})
    var tensor2 *Tensor = New_Tensor([]int{5})
    
    tensor1.data = {1,2,3,4,5}
    tensor2.data = {1,2,3,4,5}

    var cosine_similarity float64 = Cosine_Similarity(tensor1, tensor2)

-----------------------------------------------------------------------------------------------------

# Matrix Operations 
The following operations are defined for matrices, ie 2D tensors.

### Matmul()
The Matmul() function accepts two Tensor pointers and returns a pointer to a new tensor 
that is the result of the matrix multiplication of the two tensors. The function will
panic if the dimensions of the two tensors are not compatible.

    var tensor1 *Tensor = New_Tensor([]int{2,3})
    var tensor2 *Tensor = New_Tensor([]int{3,4})
    
    tensor1.data = {1,2,3,4,5,6}
    tensor2.data = {1,2,3,4,5,6,7,8,9,10,11,12}

    var tensor3 *Tensor = Matmul(tensor1, tensor2)

## Display_Matrix()
The Display_Matrix() function accepts a pointer to a tensor and prints the tensor to the
console. The function will panic if the tensor is not 2D.

    var tensor *Tensor = New_Tensor([]int{2,3})
    tensor.data = {1,2,3,4,5,6}

    Display_Matrix(tensor)

-----------------------------------------------------------------------------------------------------

# Operations for Tensors of Any Dimension
The following operations are defined for tensors of any dimension.

### Scalar_Mult_() 
The Scalar_Mult_() function accepts a pointer to a tensor and a float64 and multiplies
each element of the tensor by the float64 in place. 

    var tensor *Tensor = New_Tensor([]int{2,3})
    tensor.data = {1,2,3,4,5,6}

    Scalar_Mult_(tensor, 2)

### Add()
The Add() function accepts two Tensor pointers and returns a pointer to a new tensor
that is the result of the elementwise addition of the two tensors. The function will
panic if the dimensions of the two tensors are not compatible.

    var tensor1 *Tensor = New_Tensor([]int{2,3})
    var tensor2 *Tensor = New_Tensor([]int{2,3})
    
    tensor1.data = {1,2,3,4,5,6}
    tensor2.data = {1,2,3,4,5,6}

    var tensor3 *Tensor = Add(tensor1, tensor2)

### Subtract()
The Subtract() function accepts two Tensor pointers and returns a pointer to a new tensor
that is the result of the elementwise subtraction of the two tensors. The function will
panic if the dimensions of the two tensors are not compatible.

    var tensor1 *Tensor = New_Tensor([]int{2,3})
    var tensor2 *Tensor = New_Tensor([]int{2,3})
    
    tensor1.data = {1,2,3,4,5,6}
    tensor2.data = {1,2,3,4,5,6}

    var tensor3 *Tensor = Subtract(tensor1, tensor2)

### tensor.Reshpae()
The Reshape() method accepts a slice of integers that represent the new dimensions of
the tensor. The dimmensions must be compatible, meaning that the product of the new 
dimmensions must be equal to the product of the old dimmensions. The function will panic
if the dimmensions are not compatible. tensro.Reshape() will returns a pointer to a new
tensor with the new dimmensions.

    var tensor *Tensor = New_Tensor([]int{2,3,4})
    var tensor_reshaped *Tensor = tensor.Reshape([]int{3,8})


-----------------------------------------------------------------------------------------------------
# Statistical Operations
Statistical operations on Tensors can be broken up into two categories. The first being operations perform
some statistical operation over the all elements of the tensor, irregardless of shape. And the second being 
operations that perform some statistical operation over a specified axis of the tensor. 

# Statistical Operations over a Specified Axis

### Sum()
Sum() calculates the sum of elements in a tensor along a specified axis. This operation results in a tensor 
with one fewer dimension than the original tensor. For each position along the specified axis, there exists 
a unique combination of indices for all other axes. The function collapses the tensor by summing the values
at each unique combination of indices for the other axes, resulting in a new tensor where the dimension along 
the specified axis is removed. A pointer to the new tensor is returned.

    var tensor *Tensor = New_Tensor([]int{2,3,4})
    var tensor_summed *Tensor = tensor.Sum(1) // <-- shape: {2,4}

### Mean()
Mean() calculates the mean of elements in a tensor along a specified axis. This operation results utilizes the 
Sum() above to calculate the sum of elements along the specified axis. The sum is then divided by the number of
elements along the specified axis. This operation a pointer to a tensor with one fewer dimension than the original.

    var tensor *Tensor = New_Tensor([]int{2,3,4})
    var tensor_mean *Tensor = tensor.Mean(1) // <-- shape: {2,4}


# Statistical Operations over all Elements

### Sum_All()
Sum_All() calculates the sum of all elements in a tensor. This operation results in a float64.

    var tensor *Tensor = New_Tensor([]int{2,3,4})
    var tensor_summed float64 = tensor.Sum_All()

### Mean_All()
Mean_All() calculates the mean of all elements in a tensor. This operation results in a float64.

    var tensor *Tensor = New_Tensor([]int{2,3,4})
    var tensor_mean float64 = tensor.Mean_All()

### Var_All() 
Var_All() calculates the variance of all elements in a tensor. This operation results in a float64.

    var tensor *Tensor = New_Tensor([]int{2,3,4})
    var tensor_var float64 = tensor.Var_All()




# Code Base Conventions:

### Naming Conventions:
- Tensor pointers parameters within functions are named with capital letters starting A, B, C, etc.
- indices refers to a slice for storing a temp multidimensional index of a Tensor