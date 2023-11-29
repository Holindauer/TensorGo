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
package main

import (
	"fmt"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func main() {

	// Test Normalize_Axis()
	A := Range_Tensor([]int{3, 4}, false)

	fmt.Println("A:")
	Display_Matrix(A, false)

	fmt.Println("A_Normalized:")
	A_Normalized := A.Normalize_Axis(0) // normalize along the first axis

	Display_Matrix(A_Normalized, false)

}
