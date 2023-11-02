package main

// This source file contains the Tensor struct and functions
// related to instantiating and retrieving data from them

type Tensor struct {
	shape []int
	data  []float64 // <--- this 1D slice to store flattened tensor
}

//=============================================================================================================Accesing Data from a Tensor

// This function is used to retrieve a value from a tensor given a slice
// of the indicies at each dimension. It returns a float64 value
func (t *Tensor) Retrieve(indices []int) float64 {
	// check if each index of each dim is within the bounds of the tensor
	for i, index := range indices {
		if index >= t.shape[i] {
			panic("Retrieve() --- Index out of bounds")
		}
	}

	flattened_index := Index(indices, t.shape)
	return t.data[flattened_index]
}

// The general algorithm for computing the index of a flattened tensor from the multi dimensional indices:
// Create a slice of ints to store the strides. A stride is the number of elements in the tensor that must
// be skipped to move one index in a given dimension. Then, iterate over through each dimension of the tensor,
// multiplying the stride of that dimmension by the index of that dimension. Add the result to the flattened index.
func Index(indices []int, dims []int) int {

	// check that the number of indices matches the number of dimensions
	if len(indices) != len(dims) {
		panic("Number of indices must match number of dimensions")
	}

	strides := make([]int, len(dims)) // create a slice of ints to store the strides
	strides[len(dims)-1] = 1          // stride for the last dimension is always 1

	for i := len(dims) - 2; i >= 0; i-- { // iterate backwards through the dimensions
		strides[i] = strides[i+1] * dims[i+1] // multiply the stride of the current dimension by the size of the next dimension
		// this is because if you move one element up in dim i, then you must skip the entire
		// next dimension of the flattened tensor to get there
	}

	flattenedIndex := 0

	// iterate through tensor indices
	for i, index := range indices {
		// multiply the index by the stride of that dimension
		flattenedIndex += index * strides[i]
	}

	return flattenedIndex
}

//=============================================================================================================Creating Tensors

// TensorInitializer is an interface for initializing tensor data at each element
type TensorInitializer interface {
	ValueAt(index int) float64
}

// ConstInitializer sets a constant value for each element
type ConstInitializer struct {
	value float64
}

func (ci *ConstInitializer) ValueAt(index int) float64 {
	return ci.value
}

// RangeInitializer sets a value equal to the index for each element
type RangeInitializer struct{}

func (ri *RangeInitializer) ValueAt(index int) float64 {
	return float64(index)
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

// The following are functions for creating tensors with different initializers

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

func Range_Tensor(shape []int) *Tensor {
	initializer := &RangeInitializer{}
	return InitializeData(shape, initializer) // <--- pass initializer pointer to InitializeData()
}

// CopyInitializer copies data from another tensor
type CopyInitializer struct {
	sourceData []float64
}

func (ci *CopyInitializer) ValueAt(index int) float64 {
	return ci.sourceData[index]
}

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

// Modified Copy function
func Copy(A *Tensor) *Tensor {
	initializer := &CopyInitializer{sourceData: A.data}
	return InitializeData(A.shape, initializer)
}

// Creates Identity Matrix (Only Square Matrices)
func Eye(size int) *Tensor {
	initializer := &IdentityInitializer{size: size}
	return InitializeData([]int{size, size}, initializer)
}
