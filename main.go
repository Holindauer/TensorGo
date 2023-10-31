package main

import (
	"fmt"
)

// At this stage in the project, main is not about directing any specific processes
// It is about testing the functionality of the code in the other files.
func main() {

	// test partial on 2d tensor
	A := Range_Tensor([]int{6, 5})
	A_Partial := Partial(A, "2:3, 1:5")

	// use Display_Matrix() to display both tensors
	fmt.Println("\nA.shape = ", A.shape, "A: ")
	Display_Matrix(A)
	fmt.Println("\nA_Partial.shape: ", A_Partial.shape, "A_Partial:")
	Display_Matrix(A_Partial)

	// test tensor.Retrieved() on the Partial
	A_Partial_Retrieved := A_Partial.Retrieve([]int{0, 1})
	fmt.Println("\nA_Partial_Retrieved:", A_Partial_Retrieved)

	// test Reshape on A_Partial
	fmt.Println("\nA_Partial.shape: ", A_Partial.shape, "A_Partial after reshaping to 2x2: ")
	A_Partial_Reshaped := A_Partial.Reshape([]int{2, 2})
	Display_Matrix(A_Partial_Reshaped)

}
