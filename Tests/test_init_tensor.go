package main

// This source file contains tests of tensor creation and initialization
// functions. The tests are intended to be run with the go test command

import (
	"fmt"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
	//"time"
)

func main() {

	// Const Tensor Initialziation

	// Unbatched Ones Tensor Initialization
	fmt.Println("Unbatched Ones Tensor Initialization:\n--------------------------------------")
	ones_tensor := Ones_Tensor([]int{10, 10, 10}, false)

	fmt.Println("Expect Tensor Shape [10, 10, 10] --- Actual Shape: ", ones_tensor.Shape)
	fmt.Println("Expected Sum of Data: 1000 --- Actual Sum of Data: ", ones_tensor.Sum_All())
	fmt.Println()
	fmt.Println()

	// Batched Ones Tensor Initialization
	fmt.Println("Batched Ones Tensor Initialization:\n-------------------------------------")
	batched_ones_tensor := Ones_Tensor([]int{10, 10, 10}, true)

	fmt.Println("Expect Tensor Shape [10, 10, 10] --- Actual Shape: ", batched_ones_tensor.Shape)
	fmt.Println("Expected Sum of Data: 1000 --- Actual Sum of Data: ", batched_ones_tensor.Sum_All())
	fmt.Println()
	fmt.Println()

	// Unbatched Zeros Tensor Initialization
	fmt.Println("Unbatched Zeros Tensor Initialization:\n--------------------------------------")
	zeros_tensor := Zero_Tensor([]int{10, 10, 10}, false)

	fmt.Println("Expect Tensor Shape [10, 10, 10] --- Actual Shape: ", zeros_tensor.Shape)
	fmt.Println("Expected Sum of Data: 0 --- Actual Sum of Data: ", zeros_tensor.Sum_All())
	fmt.Println()
	fmt.Println()

	// Batched Zeros Tensor Initialization
	fmt.Println("Batched Zeros Tensor Initialization:\n-------------------------------------")
	batched_zeros_tensor := Zero_Tensor([]int{10, 10, 10}, true)

	fmt.Println("Expect Tensor Shape [10, 10, 10] --- Actual Shape: ", batched_zeros_tensor.Shape)
	fmt.Println("Expected Sum of Data: 0 --- Actual Sum of Data: ", batched_zeros_tensor.Sum_All())
	fmt.Println()
	fmt.Println()

	// Unbatched Range Tensor Initialization
	fmt.Println("Unbatched Range Tensor Initialization:\n--------------------------------------")
	range_tensor := Range_Tensor([]int{10, 10, 10}, false)

	fmt.Println("Expect Tensor Shape [10, 10, 10] --- Actual Shape: ", range_tensor.Shape)
	fmt.Println("Expected Sum of Data: 499500 --- Actual Sum of Data: ", range_tensor.Sum_All())
	fmt.Println()
	fmt.Println()

	// Batched Range Tensor Initialization
	fmt.Println("Batched Range Tensor Initialization:\n-------------------------------------")
	batched_range_tensor := Range_Tensor([]int{10, 10, 10}, true)

	fmt.Println("Expect Tensor Shape [10, 10, 10] --- Actual Shape: ", batched_range_tensor.Shape)
	fmt.Println("Expected Sum of Data: 499500 --- Actual Sum of Data: ", batched_range_tensor.Sum_All())
	fmt.Println()
	fmt.Println()

	// Copy Tensor Initialization (currently there is no batching option)
	fmt.Println("Copy Tensor Initialization:\n----------------------------")
	copy_tensor := range_tensor.Copy()

	fmt.Println("Expect Tensor Shape [10, 10, 10] --- Actual Shape: ", copy_tensor.Shape)
	fmt.Println("Expected Sum of Data: 499500 --- Actual Sum of Data: ", copy_tensor.Sum_All())
	fmt.Println()

	// Identity Matrix Tensor Initialization (currently there is no batching option)

	// Unbatched Identity Matrix Tensor Initialization (currently there is no batching option)
	fmt.Println("Unbatched Identity Matrix Tensor Initialization:\n------------------------------------------------")
	identity_matrix_tensor := Eye(10)

	Display_Matrix(identity_matrix_tensor, false)

	// Gram Matrix Tensor Initialization (currently there is no batching option)

	// Unbatched Gram Matrix Tensor Initialization (currently there is no batching option)
	fmt.Println("Unbatched Gram Matrix Tensor Initialization:\n---------------------------------------------")
	gram_matrix_tensor := Gramien_Matrix(Range_Tensor([]int{10, 10}, false))

	Display_Matrix(gram_matrix_tensor, false)

}
