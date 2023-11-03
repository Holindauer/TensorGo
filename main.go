package main

import (
	"fmt"
)

// NOTE: go over funcs/methods and be deliberate about why a func is a methods
// vs a func. Propbably make inplace methods methods and the rest funcs

// Functions I am intednding to implement are listed below

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

	//Test the concat funciton on a 2D tensor
	A := Range_Tensor([]int{2, 3})
	Display_Matrix(A)
	println("")

	B := Range_Tensor([]int{2, 3})
	Display_Matrix(B)
	println("")

	// Concatenate A and B along the first axis
	C := A.Concat(B, 0)

	// Display the concatenated tensor
	fmt.Println("Concatenated Tensor back in Main")
	Display_Matrix(C)

	// Test Transpose function on a 2D tensor
	// A := Range_Tensor([]int{2, 3})
	// Display_Matrix(A)

	// // Transpose A
	// A_T := A.Transpose([]int{1, 0})
	// Display_Matrix(A_T)

}
