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
	//"fmt"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func main() {
	// Test Case

	A := Range_Tensor([]int{3, 3, 3}, true)
	A.Data = []float64{2, 4, -2, 4, 9, -3, -2, -3, 7, 2, 4, -2, 4, 9, -3, -2, -3, 7, 2, 4, -2, 4, 9, -3, -2, -3, 7}

	x := Range_Tensor([]int{3, 3, 1}, true)
	x.Data = []float64{-1, 2, 2, -1, 2, 2, -1, 2, 2}

	Display_Matrix(A, true)
	Display_Matrix(x, true)

	b := MatMul(A, x, true)
	Display_Matrix(b, true)

	// Test Batched Gaussian Elimination
	x = Gaussian_Elimination(A, b, true)

	Display_Matrix(x, true)

	x = Gauss_Jordan_Elimination(A, b, true)
	Display_Matrix(x, true)

}
