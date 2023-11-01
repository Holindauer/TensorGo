package main

import (
	"fmt"
)

// This source file contains the Tensor struct and functions
// related to instantiating and retrieving data from them

type Tensor struct {
	shape []int
	data  []float64 // <--- this 1D slice to store flattened tensor
}

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

// This function is used to create a new tensor It takes
// in a shape and returns a Tensor pointer with that shape

func Zero_Tensor(shape []int) *Tensor {

	t := new(Tensor) //  <--- this is a pointer to a tensor
	t.shape = shape

	// compute the total number of elements in the tensor
	num_elements := 1
	for _, dim := range shape {
		num_elements *= dim
	}

	t.data = make([]float64, num_elements) // create slice of floats for data

	return t
}

func Ones_Tensor(shape []int) *Tensor {

	t := new(Tensor) //  <--- this is a pointer to a tensor
	t.shape = shape

	// compute the total number of elements in the tensor
	num_elements := 1
	for _, dim := range shape {
		num_elements *= dim
	}

	t.data = make([]float64, num_elements) // create slice of floats for data

	for i := 0; i < num_elements; i++ { // populate the tensor with 1s
		t.data[i] = 1
	}

	return t
}

// This function is used to create a new tensor where the contents
// of the flattened array range from 0 to the total number of elements
func Range_Tensor(shape []int) *Tensor {

	t := new(Tensor) //  <--- this is a pointer to a tensor
	t.shape = shape

	// compute the total number of elements in the tensor
	num_elements := 1
	for _, dim := range shape {
		num_elements *= dim
	}

	t.data = make([]float64, num_elements) // create slice of floats for data

	for i := 0; i < num_elements; i++ { // populate the tensor with the range of numbers
		t.data[i] = float64(i)
	}

	return t
}

// this function is used to display a 2D tensor as a matrix
func Display_Matrix(t *Tensor) {
	if len(t.shape) == 2 {
		// Handling 2D matrix
		for i := 0; i < t.shape[0]; i++ {
			for j := 0; j < t.shape[1]; j++ {
				fmt.Printf("%v ", t.data[i*t.shape[1]+j])
			}
			fmt.Println()
		}
	} else if len(t.shape) == 1 {
		// Handling vector
		for i := 0; i < t.shape[0]; i++ {
			fmt.Printf("%v ", t.data[i])
		}
		fmt.Println()
	} else {
		fmt.Println("Tensor must be 1D or 2D to display as matrix or vector")
	}
}

// This function checks if two tensors are of
// the same shape. It returns a boolean
func Same_Shape(A *Tensor, B *Tensor) bool {

	if len(A.shape) != len(B.shape) { // check that they have the same number of dimensions
		return false
	}

	for i := 0; i < len(A.shape); i++ { // check that each dimension is the same
		if A.shape[i] != B.shape[i] {
			return false
		}
	}

	return true
}

// This function creates a copy of a tensor and retrurns a pointer
// to the new Tensor
func Copy(A *Tensor) *Tensor {

	B := Zero_Tensor(A.shape) //  <--- returns pointer to Tensor struct

	for i := 0; i < len(A.data); i++ { // copy data from A to B
		B.data[i] = A.data[i]
	}

	return B
}

// This function creates an identity matrix, given the size of
// a single dimmension of a square matrix. A Tensor* is returned
func Eye(size int) *Tensor {

	t := Zero_Tensor([]int{size, size}) // <--- returns pointer to Tensor struct

	for i := 0; i < size; i++ {
		t_idx := Index([]int{i, i}, t.shape) // <-- populate 1s on the diagonal
		t.data[t_idx] = 1
	}

	return t
}
