# Go-LinAlg
GoLang Linear Algebra Library


The goal of this repository is to create an linear algebra library in golang that
takes advantage of concurrency. The ideas and formulas used within this repository will 
come from Gilbert Strang's Introduction to Linear Algebra.

# Documentation:
The following is an explanation of how Tensors are represened in this library, and how to use and perform operations on them.


# Tensors
The basic datatype of this library are tensors, ie multidimensional arrays. 

### Zero_Tensor() 
The Zero_Tensor() function accepts a slice of integers that represent the dimensions
of the tensor. The function returns a pointer to a tensor initialize with zeros.

    var tensor *Tensor = Zero_Tensor([]int{2,3,4})

### Range_Tensor() 
The Range_Tensor() function accepts a slice of integers that represent the dimensions
of the tensor. The function returns a pointer to a tensor initialize with the range of

    var tensor *Tensor = Range_Tensor([]int{2,3,4})

### Index()
A tensor can be indexed using the Index() function. Tensor data is represented in 
memory as a 1D contiguous array of floats, regardless of the dimensionality of the
tensor. 

The Index() function accepts a slice of integers that represent the index
of the element you want to access and a slice representing the dimensions of the
tensor. The function returns the index of the element in the 1D array.

    var tensor *Tensor = New_Tensor([]int{2,3,4})
    var index int = Index([]int{1,2,3}, []int{2,3,4})
    
    var element float64 = tensor.data[index]

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