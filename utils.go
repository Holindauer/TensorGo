package main

// utils.go contains helper functions for this projects

// Helper function for computing the product of elements in a slice
func Product(shape []int) int {
	product := 1
	for _, dim := range shape {
		product *= dim
	}
	return product
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
