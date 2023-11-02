package main

import (
	"fmt"
)

// At this stage in the project, main is not about directing any specific processes
// It is about testing the functionality of the code in the other files.
func main() {

	// Test Mean() // mean along an axis
	A := Range_Tensor([]int{900, 900, 25})
	Ones := Ones_Tensor([]int{900, 900, 25})
	B := Add(A, Ones)
	//Display_Matrix(B)

	// Test Mean() along o'th axis
	C := B.Mean(0)
	K := B.Var(0)

	// Test Sum() along o'th axis
	D := B.Sum(0)

	// Test Std() along o'th axis
	E := B.Std(0)

	fmt.Println(C.shape)
	fmt.Println(K.shape)
	fmt.Println(D.shape)
	fmt.Println(E.shape)

}
