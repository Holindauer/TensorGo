package TG

// tensor.go contains the Tensor struct and functions related to instantiating and retrieving data from them

type Tensor struct {
	Shape    []int
	Data     []float64 // <--- this 1D slice to store flattened tensor
	BoolData []bool    // <--- optional boolean data for binary tensors
	Batched  bool      // <--- optional boolean to indicate if the tensor is batched
}

//=============================================================================================================Accesing Data from a Tensor

// This function is used to retrieve a value from a tensor given a slice
// of the indicies at each dimension. It returns a float64 value
func (A *Tensor) Retrieve(indices []int) float64 {
	// check if each index of each dim is within the bounds of the tensor
	for i, index := range indices {
		if index >= A.Shape[i] {
			panic("Within Retrieve(); Index out of bounds")
		}
	}

	return A.Data[A.Index(indices)]
}

// The general algorithm for computing the index of a flattened tensor from the multi dimensional indices:
// Create a slice of ints to store the strides. A stride is the number of elements in the tensor that must
// be skipped to move one index in a given dimension. Then, iterate over through each dimension of the tensor,
// multiplying the stride of that dimmension by the index of that dimension. Add the result to the flattened index.
func (A *Tensor) Index(indices []int) int {

	// check that the number of indices matches the number of dimensions
	if len(indices) != len(A.Shape) {
		panic("Within Index(): Number of indices must match number of dimensions")
	}

	strides := make([]int, len(A.Shape)) // create a slice of ints to store the strides
	strides[len(A.Shape)-1] = 1          // stride for the last dimension is always 1

	for i := len(A.Shape) - 2; i >= 0; i-- { // decrement through the dimensions
		strides[i] = strides[i+1] * A.Shape[i+1] // multiply the stride of the current dimension by the size of the next dimension
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

// This function is a wrapper for the Index() function that allows the computation of an index from a slice of indices
// and a slice representing the dimmensions of a Tensor. This is opposed to needing an entier Tensor object.
func Index_Off_Shape(indices []int, shape []int) int {
	temp_tensor := &Tensor{Shape: shape} // 
	return temp_tensor.Index(indices)
}

// UnravelIndex converts a flat index into multi-dimensional indices based on the shape of the tensor.
// index: The flat index in the one-dimensional representation of the tensor.
// shape: The dimensions of the tensor.
func (A *Tensor) UnravelIndex(index int) []int {
	// Create a slice to store the multi-dimensional indices.
	indices := make([]int, len(A.Shape))

	// Iterate over the shape in reverse order.
	for i := len(A.Shape) - 1; i >= 0; i-- {
		// The index for the i-th dimension is the remainder of the index
		// divided by the size of the i-th dimension.
		indices[i] = index % A.Shape[i]
		// Update the index for the next iteration to be the quotient
		// of the index divided by the size of the i-th dimension.
		index = index / A.Shape[i]
	}

	// Return the calculated multi-dimensional indices.
	return indices
}

// This func is used to access a single element from a batched tensor
// NOTE: Partial() can be used to access multiple contiguous batch elements
func (A *Tensor) Extract(batch_element int) *Tensor {
	return A.Remove_Dim(0, batch_element)
}