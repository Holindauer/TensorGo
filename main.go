package main

import (
	"fmt"
)

// At this stage in the project, main is not about directing any specific processes
// It is about testing the functionality of the code in the other files.
func main() {

	// create 2x3 matrix
	matrix_1 := New_Tensor([]int{9, 3})
	matrix_2 := New_Tensor([]int{3, 7})

	// take matmul
	matmul_result := Matmul(matrix_1, matrix_2)

	fmt.Println("Matmul of matrix 1 and 2: ", matmul_result.data)

	// display result
	Display_Matrix(matmul_result)
}
