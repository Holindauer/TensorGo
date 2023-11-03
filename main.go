package main

import (
	"fmt"
)

// Functions I am intednding to implement are listed below:

// transpose --- recievs (2, 3, 1, 0) ie a new ordering of dims
// manipualtes underlying contiugous data to return a new tensor with the new ordering
// perhaps this could be implemented by adding a new member to the tensor struct called
// something like: "special_indexing" which would be able to be set to a new ordering of dims
// and then the data would be manipulated to return a new tensor with the new ordering

// Symetrix_Matrix --- R * R^T <--- Requires Transpose first

// Functions I am planning to implement are listed below:

// einsum

// various statistical functions
// mean std var sum prod

// unique elements of array

// argmax along a dimension

// argmin along a dimension

// covariance matrix computation

// normalization functions --- implement a few major strategies

// concatenate along an axis

// broadcasting

// At this stage in the project, main is not about directing any specific processes
// It is about testing the functionality of the code in the other files.
func main() {

	// Test the transpose function
	A := Range_Tensor([]int{3, 4})
	fmt.Println("A:")
	Display_Matrix(A)

	fmt.Println("A.T:")
	Display_Matrix(A.Transpose([]int{1, 0}))

}
