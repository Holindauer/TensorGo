package GLA

//import "fmt"

// This source file contains functions for initializing tensors

// TensorInitializer is an interface for initializing tensor data at each element
type TensorInitializer interface {
	ValueAt(index int) float64
}

// InitializeData creates and initializes a Tensor's data using a given TensorInitializer
func InitializeData(shape []int, initializer TensorInitializer) *Tensor {
	// create a new Tensor struct, setting its shape
	A := new(Tensor)
	A.Shape = shape
	num_elements := Product(shape) // <--- Product() is a function from utils.go
	A.Data = make([]float64, num_elements)

	for i := range A.Data {
		A.Data[i] = initializer.ValueAt(i) // <--- ValueAt() is a method of TensorInitializer
	}

	return A
}

//=============================================================================================================Tensors of Constant Values

// This is a struct that implements the TensorInitializer interface. It is used to initialize a tensor with a constant value
type ConstInitializer struct {
	value float64
}

// this is a method of ConstInitializer. It returns the val at an index of the tensor
func (ci *ConstInitializer) ValueAt(index int) float64 {
	return ci.value
}

// This is a method of ConstInitializer. It returns a tensor of the specified shape filled w/ the constant value
func (ci *ConstInitializer) Execute(shape []int) *Tensor {
	return InitializeData(shape, ci)
}

func Const_Tensor(shape []int, constant float64, batching bool) *Tensor {
	// Create a new ConstInitializer with the specified constant value
	initializer := &ConstInitializer{value: constant}
	if !batching {
		// Perform single initialization
		return InitializeData(shape, initializer)
	} else {
		// Perform batched operation
		return Batched_Initializer_Operation(initializer, shape)
	}
}

// The Zero_Tensor() and Ones_Tensor() functions are wrappers for the Const_Tensor() function
func Zero_Tensor(shape []int, batching bool) *Tensor {
	return Const_Tensor(shape, 0, batching) // <--- returns *Tensor from Const_Tensor() call
}

func Ones_Tensor(shape []int, batching bool) *Tensor {
	return Const_Tensor(shape, 1, batching) // <--- returns *Tensor from Const_Tensor() call
}

//=============================================================================================================Range Tensor

// RangeInitializer sets a value equal to the index for each element
type RangeInitializer struct{}

// This is a method of RangeInitializer. It returns the val at an index of the tensor
func (ri *RangeInitializer) ValueAt(index int) float64 {
	return float64(index)
}

// This is a method of RangeInitializer. It returns a tensor of the specified shape filled w/ the constant value
func (ri *RangeInitializer) Execute(shape []int) *Tensor {
	return InitializeData(shape, ri)
}

func Range_Tensor(shape []int, batching bool) *Tensor {
	if !batching {
		// Perform single initialization
		return InitializeData(shape, &RangeInitializer{})
	} else {
		// Perform batched operation
		return Batched_Initializer_Operation(&RangeInitializer{}, shape)
	}
}

//=============================================================================================================Copy a Tensor

// copy_tensor = tensor.Copy() creates a copy of tensor
func (A *Tensor) Copy() *Tensor {
	// Create a new tensor to store the copy.
	B := Zero_Tensor(A.Shape, false)
	copy(B.Data, A.Data) // <--- copy() is a built-in function
	return B
}

//=============================================================================================================Identity Square Matrix

// IdentityInitializer creates an identity matrix
type IdentityInitializer struct {
	size int
}

func (ii *IdentityInitializer) ValueAt(index int) float64 {
	// Determine the row and column from the index
	row := index / ii.size
	col := index % ii.size
	if row == col {
		return 1
	}
	return 0
}

// Creates Identity Matrix (Only Square Matrices)
func Eye(size int) *Tensor {
	initializer := &IdentityInitializer{size: size}
	return InitializeData([]int{size, size}, initializer)
}

//=============================================================================================================Gramien Matrix

// Gramien_Matrix(A) returns A^T * A for a 2D tensor A
// The Gramien Matrix is always symmetric
func Gramien_Matrix(A *Tensor) *Tensor {
	// Check that A 2D
	if len(A.Shape) != 2 {
		panic("Within Gramien_Matrix(): Tensor must be 2D")
	}

	return Matmul(A, A.Transpose([]int{1, 0}))
}
