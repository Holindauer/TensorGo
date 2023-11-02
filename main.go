package main

import (
	"fmt"
)

// At this stage in the project, main is not about directing any specific processes
// It is about testing the functionality of the code in the other files.
func main() {

	// Test _All functions
	A := Range_Tensor([]int{2, 3, 4})

	// Sum_All
	fmt.Println("Sum_All")
	fmt.Println(A.Sum_All())
	fmt.Println()

	// Mean_All
	fmt.Println("Mean_All")
	fmt.Println(A.Mean_All())
	fmt.Println()

	// Var_All
	fmt.Println("Var_All")
	fmt.Println(A.Var_All())
	fmt.Println()

	// Std_All
	fmt.Println("Std_All")
	fmt.Println(A.Std_All())
	fmt.Println()

}
