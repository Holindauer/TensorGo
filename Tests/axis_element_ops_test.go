package GLA

// This source file contains tests of operations that are applied along an axis that result in the axis being collapsed

import (
	"fmt"

	"testing"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func Test_Tensor_Axis_ElementOps(t *testing.T) {
	//-------------------------------------------------------------------------------------------------------------- Sum_Axis()

	fmt.Println("\nTesting Sum_Axis() unbatched...")
	A := Range_Tensor([]int{3, 3}, false)
	fmt.Println("Sum_Axis(Range_Tensor(3,3), 0):")
	fmt.Println("Original A:")
	Display_Matrix(A, false)
	fmt.Println("Sum_Axis(A, 0):")
	Display_Matrix(A.Sum_Axis(0, false), false)

	fmt.Println("\nTesting Sum_Axis() batched...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	fmt.Println("Sum_Axis(Range_Tensor(3,3,3), 0):")
	fmt.Println("Original A:")
	Display_Matrix(A, true)
	fmt.Println("\nSum_Axis(A, 0):")
	Display_Matrix(A.Sum_Axis(0, true), true)

	//-------------------------------------------------------------------------------------------------------------- Mean_Axis()

	fmt.Println("\nTesting Mean_Axis() unbatched...")
	A = Range_Tensor([]int{3, 3}, false)
	fmt.Println("Mean_Axis(Range_Tensor(3,3), 0):")
	fmt.Println("Original A:")
	Display_Matrix(A, false)
	fmt.Println("Mean_Axis(A, 0):")
	Display_Matrix(A.Mean_Axis(0, false), false)

	fmt.Println("\nTesting Mean_Axis() batched...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	fmt.Println("Mean_Axis(Range_Tensor(3,3,3), 0):")
	fmt.Println("Original A:")
	Display_Matrix(A, true)
	fmt.Println("\nMean_Axis(A, 0):")
	Display_Matrix(A.Mean_Axis(0, true), true)

	//-------------------------------------------------------------------------------------------------------------- Var_Axis()

	fmt.Println("\nTesting Var_Axis() unbatched...")
	A = Range_Tensor([]int{3, 3}, false)
	fmt.Println("Var_Axis(Range_Tensor(3,3), 0):")
	fmt.Println("Original A:")
	Display_Matrix(A, false)
	fmt.Println("Var_Axis(A, 0):")
	Display_Matrix(A.Var_Axis(0, false), false)

	fmt.Println("\nTesting Var_Axis() batched...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	fmt.Println("Var_Axis(Range_Tensor(3,3,3), 0):")
	fmt.Println("Original A:")
	Display_Matrix(A, true)
	fmt.Println("\nVar_Axis(A, 0):")
	Display_Matrix(A.Var_Axis(0, true), true)

	//-------------------------------------------------------------------------------------------------------------- Std_Axis()

	fmt.Println("\nTesting Std_Axis() unbatched...")
	A = Range_Tensor([]int{3, 3}, false)
	fmt.Println("Std_Axis(Range_Tensor(3,3), 0):")
	fmt.Println("Original A:")
	Display_Matrix(A, false)
	fmt.Println("Std_Axis(A, 0):")
	Display_Matrix(A.Std_Axis(0, false), false)

	fmt.Println("\nTesting Std_Axis() batched...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	fmt.Println("Std_Axis(Range_Tensor(3,3,3), 0):")
	fmt.Println("Original A:")
	Display_Matrix(A, true)
	fmt.Println("\nStd_Axis(A, 0):")
	Display_Matrix(A.Std_Axis(0, true), true)

}
