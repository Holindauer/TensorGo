package main

import (
	"fmt"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func main() {

	/*
		The following code is an example of how to use the Tensor-Go library to perform
		matrix multiplication as either a batched or non-batched operation.
	*/

	/*
		Below we are creating two tensors (non-batched) that are matmul compatible.
	*/

	fmt.Println("\n---------------------------")
	fmt.Println("Non-batched MatMul Example:")
	fmt.Println("---------------------------")

	// Initialize two Tensors of shape (2, 3) and (3, 7) respectively.
	var A *Tensor = RandFloat64_Tensor([]int{2, 3}, 0, 1, false)
	var B *Tensor = Range_Tensor([]int{3, 7}, false) // <--- false indicates non-batched initialization

	fmt.Println("\n\nA:")
	Display_Matrix(A, false)

	fmt.Println("\n\nB:")
	Display_Matrix(B, false)

	// Perofrm matrix multiplication
	fmt.Println("\n\nAB:")
	C := MatMul(A, B, false) // <--- false indicates non-batched operation
	Display_Matrix(C, false)

	/*
		Next we will go through the same process, only specifying the Tensors and MatMul as batched.
	*/

	fmt.Println("\n---------------------------")
	fmt.Println("Batched MatMul Example:")
	fmt.Println("---------------------------")

	// Initialize two batches of 3 Tensors each of shape (2, 3) and (3, 7) respectively.
	var A_batched *Tensor = Range_Tensor([]int{3, 2, 3}, true)
	var B_batched *Tensor = Range_Tensor([]int{3, 3, 7}, true) // <--- true indicates batched initialization

	fmt.Println("\n\nA_batched:")
	Display_Matrix(A_batched, true)

	fmt.Println("\n\nB_batched:")
	Display_Matrix(B_batched, true)

	// Perform matrix multiplication
	fmt.Println("\n\nAB_batched:")
	C_batched := MatMul(A_batched, B_batched, true) // <--- true indicates batched operation
	Display_Matrix(C_batched, true)

}
