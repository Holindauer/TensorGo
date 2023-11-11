package GLA

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
			panic("Within Retrieve(); Index out of bounds")
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
		panic("Within Index(): Number of indices must match number of dimensions")
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

// UnravelIndex converts a flat index into multi-dimensional indices based on the shape of the tensor.
// index: The flat index in the one-dimensional representation of the tensor.
// shape: The dimensions of the tensor.
func UnravelIndex(index int, shape []int) []int {
	// Create a slice to store the multi-dimensional indices.
	indices := make([]int, len(shape))

	// Iterate over the shape in reverse order.
	for i := len(shape) - 1; i >= 0; i-- {
		// The index for the i-th dimension is the remainder of the index
		// divided by the size of the i-th dimension.
		indices[i] = index % shape[i]
		// Update the index for the next iteration to be the quotient
		// of the index divided by the size of the i-th dimension.
		index = index / shape[i]
	}

	// Return the calculated multi-dimensional indices.
	return indices
}
