package main

import (
	"fmt"
)

// This source file contains the Tensor struct and functions
// related to instantiating and retrieving data from them

/*
type Tensor struct {
	shape []int
	data  []float64 // <--- this 1D slice to store flattened tensor
}
*/

// This function is used to create a new tensor It takes
// in a shape and returns a Tensor pointer with that shape

/*
func New_Tensor(shape []int) *Tensor {

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
*/

// The general algorithm for computing the index of a flattened tensor from the multi dimensional indices:
// Create a slice of ints to store the strides. A stride is the number of elements in the tensor that must
// be skipped to move one index in a given dimension. Then, iterate over through each dimension of the tensor,
// multiplying the stride of that dimmension by the index of that dimension. Add the result to the flattened index.
func Index(indices []int, dims []int) int {

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

// this function is used to display a 2D tensor as a matrix
func Display_Matrix(A *Tensor) {

	if len(A.shape) != 2 {
		panic("Tensor must be 2D to display as matrix")
	}

	for i := 0; i < A.shape[0]; i++ {
		for j := 0; j < A.shape[1]; j++ {
			A_idx := Index(A.shape, []int{i, j})
			fmt.Printf("%.1f ", A.data[A_idx])
		}
		println()
	}
}
