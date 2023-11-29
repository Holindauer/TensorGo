package GLA

// This source file contains tests of operations that are applied to all elements of a tensor in which the output  fo the operation is a single value.

import (
	"fmt"

	"testing"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func Test_Tensor_All_Element_Ops(t *testing.T) {
	//-------------------------------------------------------------------------------------------------------------- Sum_All()

	fmt.Println("\nTesting Sum_All() unbatched...")
	A := Range_Tensor([]int{3, 3}, false)
	fmt.Println("Sum_All(Range_Tensor(3,3)):", A.Sum_All())

	fmt.Println("\nTesting Sum_All() batched...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	fmt.Println("Sum_All(Range_Tensor(3, 3, 3)):", A.Sum_All())

	//-------------------------------------------------------------------------------------------------------------- Mean_All()

	fmt.Println("\nTesting Mean_All() unbatched...")
	A = Range_Tensor([]int{3, 3}, false)
	fmt.Println("Mean_All(Range_Tensor(3,3)):", A.Mean_All())

	fmt.Println("\nTesting Mean_All() batched...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	fmt.Println("Mean_All(Range_Tensor(3,3,3)):", A.Mean_All())

	//-------------------------------------------------------------------------------------------------------------- Var_All()

	fmt.Println("\nTesting Var_All() unbatched...")
	A = Range_Tensor([]int{3, 3}, false)
	fmt.Println("Var_All(Range_Tensor(3,3)):", A.Var_All())

	fmt.Println("\nTesting Var_All() batched...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	fmt.Println("Var_All(Range_Tensor(3,3,3)):", A.Var_All())

	//-------------------------------------------------------------------------------------------------------------- Std_All()

	fmt.Println("\nTesting Std_All() unbatched...")
	A = Range_Tensor([]int{3, 3}, false)
	fmt.Println("Std_All(Range_Tensor(3,3)):", A.Std_All())

	fmt.Println("\nTesting Std_All() batched...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	fmt.Println("Std_All(Range_Tensor(3,3,3)):", A.Std_All())

}
