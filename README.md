# Go-LinAlg
GoLang Linear Algebra Library


The goal of this repository is to create an linear algebra library in golang that
takes advantage of concurrency. The ideas and formulas for this repository will 
come from Gilbert Strang's Introduction to Linear Algebra.

## Documentation:

### Tensors
The basic datatype of this library are tensors, ie multidimensional array. 

The New_Tensor() function accepts a slice of integers that represent the dimensions
of the tensor. The function returns a pointer to a tensor initialize with zeros.

    var tensor *Tensor = New_Tensor([]int{2,3,4})

### Indexing
The tensor can be indexed using the Index() function. Tensor data is represented in 
memory as a 1D contiguous array of floats, regardless of the dimensionality of the
tensor. 

The Index() function accepts a slice of integers that represent the index
of the element you want to access and a slice representing the dimensions of the
tensor. The function returns the index of the element in the 1D array.

    var tensor *Tensor = New_Tensor([]int{2,3,4})
    var index int = Index([]int{1,2,3}, []int{2,3,4})
    
    var element float64 = tensor.data[index]

### Dot Product
The Dot() function accepts two tensors and returns the dot product of the two tensors.
The function will panic if the dimensions of the two tensors are not compatible.

    var tensor1 *Tensor = New_Tensor([]int{5})
    var tensor2 *Tensor = New_Tensor([]int{5})
    
    tensor1.data = {1,2,3,4,5}
    tensor2.data = {1,2,3,4,5}

    var length_squared float64 = Dot(tensor1, tensor2)

Relatedly, the Norm() function accepts a tensor and returns the norm of the tensor.

    var tensor *Tensor = New_Tensor([]int{5})
    tensor.data = {1,2,3,4,5}

    var norm float64 = Norm(tensor)         
    length = math.Sqrt(Dot(tensor, tensor))  // <--- equivalent to Norm()

### Matrix Multiplication
The MatMul() function accepts two Tensor pointers and returns a pointer to a new tensor 
that is the result of the matrix multiplication of the two tensors. The function will
panic if the dimensions of the two tensors are not compatible.

    var tensor1 *Tensor = New_Tensor([]int{2,3})
    var tensor2 *Tensor = New_Tensor([]int{3,4})
    
    tensor1.data = {1,2,3,4,5,6}
    tensor2.data = {1,2,3,4,5,6,7,8,9,10,11,12}

    var tensor3 *Tensor = MatMul(tensor1, tensor2)
