package main

import (
	"fmt"
)

// At this stage in the project, main is not about directing any specific processes
// It is about testing the functionality of the code in the other files.
func main() {

	// test partial on 2d tensor
	A := Range_Tensor([]int{5, 4})
	A_Partial := Partial(A, ":3, 2:")

	// use Display_Matrix() to display both tensors
	fmt.Println("\nA:")
	Display_Matrix(A)
	fmt.Println("\nA_Partial:")
	Display_Matrix(A_Partial)

	// test tensor.Retrieved() on the Partial
	A_Partial_Retrieved := A_Partial.Retrieve([]int{2, 1})
	fmt.Println("\nA_Partial_Retrieved:", A_Partial_Retrieved)

}
