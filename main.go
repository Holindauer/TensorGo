package main

///import "fmt"

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

	// Test the ArgMax function
	A := Zero_Tensor([]int{3, 3})
	// 2, 4, -2, 4, 9, -3, -2, -3, 7
	A.data = []float64{2, 4, -2, 4, 9, -3, -2, -3, 7}
	Display_Matrix(A)

	b := Zero_Tensor([]int{3})
	b.data = []float64{2, 8, 10}
	Display_Matrix(b)

	// Test Gaussian Elimination
	x := Gaussian_Elimination(A, b)

	Display_Matrix(x)

}
