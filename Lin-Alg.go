package main

type Tensor struct {
	shape []int
	data  []float64 // <--- this 1D slice is used to store the data of the tensor
	//      the indices of the data are mapped to the indices of the tensor
}

// This function is used to create a new tensor
// It takes in a shape and returns a tensor with that shape
func New_Tensor(shape []int) *Tensor {
	// Create a new tensor
	t := new(Tensor)

	// Set the shape of the tensor
	t.shape = shape
	// Create a new slice of floats to store the data
	t.data = make([]float64, len(shape))
	// Return the tensor
	return t
}

/*
The general algorithm for computing the index of a flattened
tensor from the multi dimensional indices is as follows:
1.) Create a slice of ints to store the strides. A stride is the number of elements

	in the tensor that must be skipped to move one index in a given dimension.

2.) Iterate over through each dimension of the tensor, multiplying the stride of that

	dimmension by the index of that dimension. Add the result to the flattened index.
*/
func getFlattenedIndex(indices []int, dims []int) int {

	strides := make([]int, len(dims)) // create a slice of ints to store the strides
	strides[len(dims)-1] = 1          // stride for the last dimension is always 1

	for i := len(dims) - 2; i >= 0; i-- { // iterate backwards through the dimensions
		strides[i] = strides[i+1] * dims[i+1] // multiply the stride of the current dimension by the size of the next dimension
		// this is because if you move one element up in dim i, then you must skip the entire
		// next dimension of the flattened tensor to get there
	}

	flattenedIndex := 0

	for i, index := range indices { // iterate through tensor indices
		flattenedIndex += index * strides[i] // multiply the index by the stride of that dimension
	}

	return flattenedIndex
}
