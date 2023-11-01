package main

import (
	"fmt"
)

// At this stage in the project, main is not about directing any specific processes
// It is about testing the functionality of the code in the other files.
func main() {

	// Creating a 5x5 matrix
	matrix := Range_Tensor([]int{5, 5})
	fmt.Println("Original 5x5 Matrix:")
	Display_Matrix(matrix)
	fmt.Println()

	// Sum along axis 0
	sumAxis0 := matrix.Sum(0)
	fmt.Println("Sum along axis 0:")
	Display_Matrix(sumAxis0)
	fmt.Println()

	// Sum along axis 1
	sumAxis1 := matrix.Sum(1)
	fmt.Println("Sum along axis 1:")
	Display_Matrix(sumAxis1)
	fmt.Println()

	// Test Mean()

	// Creating a 5x5 matrix
	matrix = Range_Tensor([]int{5, 5})
	fmt.Println("Original 5x5 Matrix:")
	Display_Matrix(matrix)
	fmt.Println()

	// Mean along axis 0
	meanAxis0 := matrix.Mean(0)
	fmt.Println("Mean along axis 0:")
	Display_Matrix(meanAxis0)
	fmt.Println()

	// Mean along axis 1
	meanAxis1 := matrix.Mean(1)
	fmt.Println("Mean along axis 1:")
	Display_Matrix(meanAxis1)
	fmt.Println()

}
