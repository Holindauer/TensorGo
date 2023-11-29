package GLA

// This source file contains tests of operations that are applied to all elements of a tensor in which the output  fo the operation is a single value.

import (
	"fmt"

	"testing"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func Test_Tensor_All_Inplace_Ops(t *testing.T) {

	fmt.Println("\nTesting all_inplace_ops...")

	//-------------------------------------------------------------------------------------------------------------- Add()

	fmt.Println("\nTesting Add() unbatched...")
	A := Ones_Tensor([]int{3, 3}, false)
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

	//-------------------------------------------------------------------------------------------------------------- Normalize()

	fmt.Println("\nTesting Normalize() unbatched...")

	A = Range_Tensor([]int{10, 10}, false)

	fmt.Println("A:")
	Display_Matrix(A, false)

	// Test Normalize
	A_Normalize := A.Normalize(false)

	fmt.Println(" A_Normalized:")
	Display_Matrix(A_Normalize, false)

	fmt.Println("\nTesting Normalize() batched...")

	A = Range_Tensor([]int{2, 10, 10}, true)

	fmt.Println("A:")
	Display_Matrix(A.Normalize(false), true)

}
