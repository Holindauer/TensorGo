package GLA

// This source file contains tests of operationss that are applied to all elements of a tensor (not axis-wise).

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

	//-------------------------------------------------------------------------------------------------------------- Add()

	fmt.Println("\nTesting Add() unbatched...")
	A = Ones_Tensor([]int{3, 3}, false)
	A_added := Add(A, A)
	fmt.Println("Add(Ones_Tensor(3,3), Ones_Tensor(3,3)):")
	Display_Matrix(A_added, false)

	fmt.Println("\nTesting Add() batched...")
	A = Ones_Tensor([]int{3, 3, 3}, true)
	A_added = Add(A, A)
	fmt.Println("Add(Ones_Tensor(3,3,3), Ones_Tensor(3,3,3)):")
	Display_Matrix(A_added, true)

	//-------------------------------------------------------------------------------------------------------------- Subtract()

	fmt.Println("\nTesting Subtract() unbatched...")
	A = Ones_Tensor([]int{3, 3}, false)
	A_subtracted := Subtract(A, A)
	fmt.Println("Subtract(Ones_Tensor(3,3), Ones_Tensor(3,3)):")
	Display_Matrix(A_subtracted, false)

	fmt.Println("\nTesting Subtract() batched...")
	A = Ones_Tensor([]int{3, 3, 3}, true)
	A_subtracted = Subtract(A, A)
	fmt.Println("Subtract(Ones_Tensor(3,3,3), Ones_Tensor(3,3,3)):")
	Display_Matrix(A_subtracted, true)

	//-------------------------------------------------------------------------------------------------------------- Scalar_Mult()

	fmt.Println("Testing Scalar_Mult() unbatched...")
	A = Ones_Tensor([]int{3, 3}, false)
	A_scalar_mult := A.Scalar_Mult_(2)
	fmt.Println("\nScalar_Mult(Ones_Tensor(3,3), 2):")
	Display_Matrix(A_scalar_mult, false)

	fmt.Println("\nTesting Scalar_Mult() batched...")
	A = Ones_Tensor([]int{3, 3, 3}, true)
	A_scalar_mult = A.Scalar_Mult_(2)
	fmt.Println("\nScalar_Mult(Ones_Tensor(3,3,3), 2):")
	Display_Matrix(A_scalar_mult, true)
}
