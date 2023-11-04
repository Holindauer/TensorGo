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

	// Test Concat() on a 2d tensor in both axes
	t1 := Range_Tensor([]int{2, 5})
	t2 := Range_Tensor([]int{2, 5})

	Display_Matrix(t1)
	fmt.Println("")
	Display_Matrix(t2)
	fmt.Println("")

	t3 := t1.Concat(t2, 0)
	Display_Matrix(t3)
	fmt.Println("")

	t4 := t1.Concat(t2, 1)
	Display_Matrix(t4)

}
