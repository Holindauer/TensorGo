package main

// This source file contains functions for initializing tensors

// TensorInitializer is an interface for initializing tensor data at each element
type TensorInitializer interface {
	ValueAt(index int) float64
}

// InitializeData creates and initializes a Tensor's data using a given TensorInitializer
func InitializeData(shape []int, initializer TensorInitializer) *Tensor {
	// create a new Tensor struct, setting its shape
	A := new(Tensor)
	A.shape = shape
	num_elements := Product(shape) // <--- Product() is a function from utils.go
	A.data = make([]float64, num_elements)

	for i := range A.data {
		A.data[i] = initializer.ValueAt(i) // <--- ValueAt() is a method of TensorInitializer
	}

	return A
}

//=============================================================================================================Tensors of Constant Values

// ConstInitializer is used to set a constant value for an element
type ConstInitializer struct {
	value float64
}

// ValueAt returns the constant value
func (ci *ConstInitializer) ValueAt(index int) float64 {
	return ci.value
}

// Const_Tensor creates a tensor of a given shape with a constant value
func Const_Tensor(shape []int, constant float64) *Tensor {
	initializer := &ConstInitializer{value: constant}
	return InitializeData(shape, initializer)
}

func Zero_Tensor(shape []int) *Tensor {
	return Const_Tensor(shape, 0) // <--- returns *Tensor from Const_Tensor() call
}

func Ones_Tensor(shape []int) *Tensor {
	return Const_Tensor(shape, 1) // <--- returns *Tensor from Const_Tensor() call
}

//=============================================================================================================Range Tensor

// RangeInitializer sets a value equal to the index for each element
type RangeInitializer struct{}

func (ri *RangeInitializer) ValueAt(index int) float64 {
	return float64(index)
}

func Range_Tensor(shape []int) *Tensor {
	initializer := &RangeInitializer{}
	return InitializeData(shape, initializer) // <--- pass initializer pointer to InitializeData()
}

//=============================================================================================================Copy a Tensor

// copy_tensor = tensor.Copy() creates a copy of tensor
func (A *Tensor) Copy() *Tensor {
	// Create a new tensor to store the copy.
	B := Zero_Tensor(A.shape)
	copy(B.data, A.data) // <--- copy() is a built-in function
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
	if len(A.shape) != 2 {
		panic("Within Gramien_Matrix(): Tensor must be 2D")
	}

	return Matmul(A, A.Transpose([]int{1, 0}))
}
