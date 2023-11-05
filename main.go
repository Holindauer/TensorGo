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

	// Test the Extend() funciton on a 1D Zero_Tensor
	tensor := Range_Tensor([]int{7})

	fmt.Println("The tensor is:", tensor)

	// Test the Extend() funciton on tensor

	fmt.Println("The extended tensor is:")
	extended_tensor := tensor.Extend(6)

	Display_Matrix(extended_tensor)

}
