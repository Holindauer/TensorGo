# Go-LinAlg
GoLang Linear Algebra Library

-----------------------------------------------------------------------------------------------------

The goal of this repository is to create a linear algebra library in GoLang that is easy to use and understand.
Many of the ideas and concepts implemented in this repository come from Introduction to Linear Algebra by Gilbert Strang
and adaptations from the NumPy library.

# Documentation:

-----------------------------------------------------------------------------------------------------

# Tensors
This library represents Tensors (n-dimmensional arrays) as a 1D slice of float64 values stored contiguosly in memory.  
Multidimmensionality is simulated using a stride based indexing schema. A Tensor struct contains two members:

    - Data: A 1D slice of float64 values
    - Shape: A 1D slice of int values representing the multi-dimmensional shape of the tensor

-----------------------------------------------------------------------------------------------------

# Tensor Initialization
Tensor initialization functions in this library always return a pointer to a Tensor struct.  The following functions
are used to create different types of Tensors:

### Const_Tensor(), Zero_Tensor(), Ones_Tensor()
The Const_Tensor() function accepts a slice of ints representing the shape of the Tensor and a float64 value to fill its data member with.
Range_Tensor and Ones_Tensors() call Const_Tensor() internally, filling the data member with the float64 value their names suggest. 

    var A *Tensor = Const_Tensor([]int{2, 2, 8}, 1.7) // <--- Creates a 2x2x8 Tensor filled with 1.7

    var B *Tensor = Zero_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with 0.0 
    var C *Tensor = Ones_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with 1.0

### Range_Tensor()
The Range_Tensor() function accepts a slice of ints representing the shape of the Tensor and fills the contiguous memory with float64 values
incrementing up the indicies from 0 to len(tensor.data)-1

    var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63

### Copy()
The Copy() function accepts a pointer to a Tensor struct and returns a pointer to a new Tensor struct with the same shape and data values as the input Tensor.

    var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
    var B *Tensor = Copy(A)                      // <--- Creates a new Tensor with the same shape and data values as A

### Eye()
The Eye() function accepts an integer argument represeting the shape of a square matrix. It returns a pointer to a 2D Tensor struct containing an identity matrix.

    var A *Tensor = Eye(3) // <--- Creates a 3x3 identity matrix

### Gramien_Matrix()
The Gramient_Matrix() function accepts a 2D Tensor and returns a pointer to a Tensor struct that is the Matrix Multiplication of the input Tensor and its transpose.

    var A *Tensor = Range_Tensor([]int{8, 7}) // <--- Creates a 8x7 Tensor filled with values from 0 to 55
    var B *Tensor = Gramien_Matrix(A)         // <--- Creates a 8x8 Gramien Matrix

-----------------------------------------------------------------------------------------------------

# Tensor Indexing
Tensor data is stored in memory as a 1D slice of float64 values. Multidimmensionality is simulated using the follwing indexing functions. These functions implement a stride based indexing schema. 
This means that for a slice of ints representing the multidimensional index of a tensor, the index of the flat 1D slice that corresponds to the multidimensional index is computed and then can be used
to simply index the 1D slice. The intuition is that for a flattned multidimmensional array, in order to reach the next element of a given dimmension, you must "skip" over all the values of the high dimmension above it (contiguously stored values that is).

### Retrieve()
The retireve method acts on a Tensor struct and accepts a slice of ints representing an indexing of that Tensor. The method returns the value at that indexing.
    
        var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
        var B float64 = A.Retrieve([]int{1, 1, 1})   // <--- B = 29

### Index()
The Index() function is called internally within Retrieve() to calculate the flattened index from a multidimmensional array. However it can also be called on its own. It accepts 
a slice of ints representing the multidimmensional indexing meant to be retrieved from the Tensor. As well as the shape of the Tensor being indexed. It returns an integer of the flat index.

        var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
        var idx int = Index([]int{1, 1, 1}, A.shape) // <--- idx = 29

### Unravel_Index()
If you find yourself in a situation where you need to compute the multidimmensional indexing of a Tensor from its flattened index, you can use the Unravel_Index() function. It accepts an integer representing the flattened index
and a slice of ints representing the shape of the Tensor. It returns a slice of ints representing the multidimmensional indexing of the Tensor.

        var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
        var idx int = Index([]int{1, 1, 1}, A.shape) // <--- idx = 29
        var idxs []int = Unravel_Index(idx, A.shape) // <--- idxs = []int{1, 1, 1}

-----------------------------------------------------------------------------------------------------

# Tensor Shape Operations
The following functions are used to manipulate the shape of a Tensor.

### Partial()
The Partial method is used to retrieve a portion of a Tensor. It acts on a Tensor struct and accepts a string containing a python stye slice notation indexing of the Tensor. It returns a pointer to a new Tensor struct containing the portion of the Tensor specified by the slice notation. The Bounds of the slice notation are exlusive, to index the end of the Tensor, use the ":" character. The ":" character can also be used to index the entire Tensor along a given dimmension.

        var A *Tensor = Range_Tensor([]int{8, 8, 8, 8}) // <--- Creates an 8x8x8x8 Tensor 
        var B *Tensor = A.Partial(":, 4:6, :6, 2:")     // <--- Creates a 8x2x6x6 Tensor

### Reshape()
The Reshape() method Copys the data from the Tensor it acts on into the shape specified by the slice of ints passed in as an argument. The product of the shape must be equal to the length of the data. It returns a pointer to a new Tensor struct with the new shape.
    
            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B *Tensor = A.Reshape([]int{4, 4})       // <--- Creates a 4x4 Tensor filled with values from 0 to 15

### Transpose()
The Transpose method acts on a Tensor and accepts a slice of ints representing an intended reordering of the dimmensions of the Tensor. Under the hood, creates a new Tensor struct of which its contiguous data has been reorganized to match the dimmensions of the swapped dimmensions. It returns a pointer to a new Tensor struct with the new shape.

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B *Tensor = A.Transpose([]int{2, 0, 1})  // <--- Creates a 8x2x2 Tensor the data is no longer 0 to 63

### Concat()
The Concat() method acts on a Tensor by concatenating a Tensor argument along a specified dimmension. It returns a pointer to a new Tensor struct with the new shape. 

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var C *Tensor = A.Concat(B, 2)               // <--- Creates a 2x2x16 Tensor

Under the hood, the Tensor is Transposed such that the axis of concatenation is swapped with the first dimmension (if needed) then the contiguos data of the arg Tensor is appeneded contiguously to the data of the Tensor it acts on. Finally the shape is updated to reflect the new shape.

### Extend_Dim()
The Extened_Dim() method acts on a Tensor by extending the length of a specified dimmension by a specified amount. The new values along that dimmension are initialized to zero. 

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B *Tensor = A.Extend_Dim(2, 4)           // <--- Creates a 2x2x12 Tensor with the new 2x2x4 data initialized to 0.0

### Extend_Shape()
The Extend_Shape() method acts on a Tensor by extneding the number of dimmension of the Tensor by 1. This added dim is appended to the end of the original shape. This new dimmension has the elements specified by the num_elements integer argument. For each element of the new dim, it can be thought of that a new 'state' of the original Tensor shape has been created. By deault, this 'state' is the same as the original Tensor. This becomes more dramatic with higher dimmensional Tensors.

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B *Tensor = A.Extend_Shape(4)            // <--- Creates a 2x2x8x4 Tensor with the new 2x2x8 data initialized to 0.0
                                                         //      Each 2x2x8 'state' across the 0'th dim is the same as the original Tensor 


-----------------------------------------------------------------------------------------------------

# Vector Operations
The following Tensor operations are used to manipulate Tensors that are vectors (1D Tensors).

### dot()
The dot() function accepts two 1D tensor and returns a float64 value that is the dot product of the two vectors. Tensors must be the same shape vectors.

            var A *Tensor = Range_Tensor([]int{8}) // <--- Creates a 8x1 Tensor filled with values from 0 to 7
            var B *Tensor = Range_Tensor([]int{8}) // <--- Creates a 8x1 Tensor filled with values from 0 to 7
            var C float64 = dot(A, B)              // <--- C = 140


### Norm() 
The Norm() function returns the float64 norm of two vector Tensors. 

            var A *Tensor = Range_Tensor([]int{8}) 
            var B float64 = Norm(A)                

### Unit()
The Unit() function returns a pointer to a new Tensor struct that is the unit vector of the input vector Tensor.

            var A *Tensor = Range_Tensor([]int{8}) 
            var B *Tensor = Unit(A)

## Check_Perpendicular()
The Check_Perpendicular() function returns a boolean value indicating if two vector Tensors are perpendicular.

            var A *Tensor = Range_Tensor([]int{8}) 
            var B *Tensor = Range_Tensor([]int{8}) 
            var C bool = Check_Perpendicular(A, B) // <--- C = false

### Cosine_Similarity()
The Cosine_Similarity() function returns a float64 value indicating the cosine similarity of two vector Tensors.

            var A *Tensor = Range_Tensor([]int{8}) 
            var B *Tensor = Range_Tensor([]int{8}) 
            var C float64 = Cosine_Similarity(A, B) // <--- C = 1.0

### Outer_Product()
The Outer_Product() function returns a pointer to a new Tensor struct that is the outer product of two vector Tensors.

            var A *Tensor = Range_Tensor([]int{8}) 
            var B *Tensor = Range_Tensor([]int{8}) 
            var C *Tensor = Outer_Product(A, B)     // <--- C is a 8x8 Tensor

-----------------------------------------------------------------------------------------------------

# Matrix Operations

### Matmul()
The Matmul() function accepts two 2D Tensor structs and returns a pointer to a new Tensor struct that is the matrix multiplication of the two input Tensors. The shape of the two input Tensors must be compatible for matrix multiplication.

            var A *Tensor = Range_Tensor([]int{8, 7}) // <--- Creates a 8x7 Tensor filled with values from 0 to 55
            var B *Tensor = Range_Tensor([]int{7, 8}) // <--- Creates a 7x8 Tensor filled with values from 0 to 55
            var C *Tensor = Matmul(A, B)              // <--- Creates a 8x8 Tensor

### Display_Matrix()

The Display_Matrix() function accepts a 2D Tensor struct and prints the values of the Tensor in a matrix format.

            var A *Tensor = Range_Tensor([]int{8, 7}) // <--- Creates a 8x7 Tensor filled with values from 0 to 55
            Display_Matrix(A)                         // <--- Prints the values of A in a matrix format

-----------------------------------------------------------------------------------------------------

# Operations Along an Axis
The following methods perform operation on Tensors along a specified axis. This can be thought of intuitively as performing an elementwise operation along the 'state' of the Tensor outlined by the other dimensions of 
the Operand Tensor at a given element along the axis of operation. For example, if you have a 3D Tensor of shape 2x2x8 and you perform an operation along the 0'th axis, you can think of this as performing an operation on the two 2x8 'states' of the Tensor which sit along that axis. The result is a 2x8 Tensor, which is 1 dimmension less because the axis of operation has been collapsed into the other dimmensions in order to perform the operation.

### Sum_Axis()
The Sum_Axis() method acts on a Tensor and accepts an integer representing the axis along which the sum is to be performed. It returns a pointer to a new Tensor struct with the new shape. 

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B *Tensor = A.Sum_Axis(0)                // <--- Creates a 2x8 Tensor

### Mean_Axis()
The Mean_Axis() method acts on a Tensor and accepts an integer representing the axis along which the mean is to be performed. It returns a pointer to a new Tensor struct with the new shape. 

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B *Tensor = A.Mean_Axis(0)               // <--- Creates a 2x8 Tensor

### Var_Axis()
The Var_Axis() method acts on a Tensor and accepts an integer representing the axis along which the variance is to be performed. It returns a pointer to a new Tensor struct with the new shape. 

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B *Tensor = A.Var_Axis(0)                // <--- Creates a 2x8 Tensor

### Std_Axis()
The Std_Axis() method acts on a Tensor and accepts an integer representing the axis along which the standard deviation is to be performed. It returns a pointer to a new Tensor struct with the new shape. 

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B *Tensor = A.Std_Axis(0)                // <--- Creates a 2x8 Tensor

-----------------------------------------------------------------------------------------------------

# Operations Across All Elements
The following function/methods act upon all elements of a Tensor at once. There are two types of operations that fall into this category. That being ones that return a single scalar value and ones that perform an elementwise operation of a Tensor and return a new Tensor of the same shape.

# Operations that Return a Scalar

### Sum_All()
The Sum_All() method adds up all elements in a Tensor into float64 value

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B float64 = A.Sum_All()                  // <--- B = 2016

### Mean_All()
The Mean_All() method computes the mean of all elements in a Tensor into float64 value

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B float64 = A.Mean_All()                 // <--- B = 7.875

### Var_All()
The Var_All() method computes the variance of all elements in a Tensor into float64 value

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B float64 = A.Var_All()                  // <--- B = 546.0

### Std_All()
The Std_All() method computes the standard deviation of all elements in a Tensor into float64 value

            var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor filled with values from 0 to 63
            var B float64 = A.Std_All()                  // <--- B = 23.366642891095847

# Operations that Return a Tensor

### Add()
The Add() function performs an elementwise addition of two Tensors, returning a pointer to a new Tensor of their Sum.
    
                var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor
                var B *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor
                var C *Tensor = Add(A, B)                    // <--- Creates a 2x2x8 Tensor

### Subtract()
The Subtract() function performs an elementwise subtraction of two Tensors, returning a pointer to a new Tensor of their Difference.
    
                var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor
                var B *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor
                var C *Tensor = Subtract(A, B)               // <--- Creates a 2x2x8 Tensor

### Scalar_Mult()
The Scalar_Mult() method acts on a Tensor and accepts a float64 value. It returns a pointer to a new Tensor struct with the new shape. 

                var A *Tensor = Range_Tensor([]int{2, 2, 8}) // <--- Creates a 2x2x8 Tensor
                var B *Tensor = A.Scalar_Mult(2.0)           // <--- Creates a 2x2x8 Tensor

