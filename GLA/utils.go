package GLA

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

	if len(A.Shape) != len(B.Shape) { // check that they have the same number of dimensions
		return false
	}

	for i := 0; i < len(A.Shape); i++ { // check that each dimension is the same
		if A.Shape[i] != B.Shape[i] {
			return false
		}
	}

	return true
}

// This function is used to create a slice of integer indicies from 0 to n -1 and then have the 0'th and n - 1'th indicies swapped
// This is used to reorder the indicies of a tensor to reorder the contiguous memory of a tensor
func Indicies_First_Last_Swapped(n int) []int {
	indicies := make([]int, n)
	for i := range indicies {
		indicies[i] = i
	}
	indicies[0] = n - 1
	indicies[n-1] = 0
	return indicies
}
