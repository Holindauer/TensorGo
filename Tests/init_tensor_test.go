package GLA

// This source file contains tests of tensor creation and initialization
// functions. The tests are intended to be run with the go test command

import (
	"fmt"

	"testing"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func Test_Tensor_Init(t *testing.T) {

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
	fmt.Println()

	// Unbatched Identity Matrix Tensor Initialization
	fmt.Println("Unbatched Identity Matrix Tensor Initialization:\n------------------------------------------------")
	identity_matrix_tensor := Eye([]int{10, 10}, false)

	Display_Matrix(identity_matrix_tensor, false)
	fmt.Println()
	fmt.Println()

	// Batched Identity Matrix Tensor Initialization
	fmt.Println("Batched Identity Matrix Tensor Initialization:\n-----------------------------------------------")
	batched_identity_matrix_tensor := Eye([]int{3, 10, 10}, true)

	Display_Matrix(batched_identity_matrix_tensor, true)
	fmt.Println()
	fmt.Println()

	// Gram Matrix Tensor Initialization (currently there is no batching option)

	// Unbatched Gram Matrix Tensor Initialization (currently there is no batching option)
	fmt.Println("Unbatched Gram Matrix Tensor Initialization:\n---------------------------------------------")
	gram_matrix_tensor := Gram(Range_Tensor([]int{10, 10}, false), false)

	Display_Matrix(gram_matrix_tensor, false)
	fmt.Println()
	fmt.Println()

	// Batched Gram Matrix Tensor Initialization (currently there is no batching option)
	fmt.Println("Batched Gram Matrix Tensor Initialization:\n---------------------------------------------")
	batched_gram_matrix_tensor := Gram(Range_Tensor([]int{3, 10, 10}, true), true)

	Display_Matrix(batched_gram_matrix_tensor, true)
	fmt.Println()
	fmt.Println()

	// Unbatched Random Tensor Initialization
	fmt.Println("Unbatched Random Tensor Initialization:\n--------------------------------------")
	random_tensor := RandFloat_Tensor([]int{10, 10, 10}, 0, 1, false)

	fmt.Println("Expect Tensor Shape [10, 10, 10] --- Actual Shape: ", random_tensor.Shape)
	fmt.Println()
	fmt.Println()

	// Batched Random Tensor Initialization
	fmt.Println("Batched Random Tensor Initialization:\n-------------------------------------")
	batched_random_tensor := RandFloat_Tensor([]int{3, 10, 10}, 0, 1, true)

	fmt.Println("Expect Tensor Shape [10, 10, 10] --- Actual Shape: ", batched_random_tensor.Shape)
	fmt.Println()
	Display_Matrix(batched_random_tensor, true)
	fmt.Println()
}
