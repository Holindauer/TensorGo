package GLA

// This source file contains tests of operats that are applied along an axis

import (
	"fmt"

	"testing"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func Test_Elimination(t *testing.T) {

	fmt.Println("Testing elimination.go")

	//-------------------------------------------------------------------------------------------------------------- Gaussian_Elimination()

	fmt.Println("Testing Gaussian_Elimination() unbatched")

	// creat an A and x, multiply them to get b
	A := Zero_Tensor([]int{3, 3}, false)
	A.Data = []float64{2, 4, -2, 4, 9, -3, -2, -3, 7}
	x := Zero_Tensor([]int{3, 1}, false)
	x.Data = []float64{-1, 2, 2}
	b := MatMul(A, x, false)

	fmt.Println("\nA:")
	Display_Matrix(A, false)
	fmt.Println("\nb:")
	Display_Matrix(b, false)
	fmt.Println("\nx:")
	Display_Matrix(x, false)

	// solve for x
	x_solved := Gaussian_Elimination(A, b, false)
	fmt.Println("x_solved by Gaussian_Elimination() unbatched:")
	Display_Matrix(x_solved, false)

	fmt.Println("Testing Gaussian_Elimination() batched")
	A = Zero_Tensor([]int{3, 3, 3}, true)
	A.Data = []float64{2, 4, -2, 4, 9, -3, -2, -3, 7, 2, 4, -2, 4, 9, -3, -2, -3, 7, 2, 4, -2, 4, 9, -3, -2, -3, 7}
	x = Zero_Tensor([]int{3, 3, 1}, true)
	x.Data = []float64{-1, 2, 2, -1, 2, 2, -1, 2, 2}
	b = MatMul(A, x, true)

	fmt.Println("\nA:")
	Display_Matrix(A, true)
	fmt.Println("\nb:")
	Display_Matrix(b, true)
	fmt.Println("\nx:")
	Display_Matrix(x, true)

	// solve for x
	x_solved = Gaussian_Elimination(A, b, true)
	fmt.Println("x_solved by Gaussian_Elimination() abtched:")
	Display_Matrix(x_solved, true)

	//-------------------------------------------------------------------------------------------------------------- Gauss_Jordan_Elimination()

	fmt.Println("Testing Gauss_Jordan_Elimination() unbatched")

	// creat an A and x, multiply them to get b
	A = Zero_Tensor([]int{3, 3}, false)
	A.Data = []float64{2, 4, -2, 4, 9, -3, -2, -3, 7}
	x = Zero_Tensor([]int{3, 1}, false)
	x.Data = []float64{-1, 2, 2}
	b = MatMul(A, x, false)

	fmt.Println("\nA:")
	Display_Matrix(A, false)
	fmt.Println("\nb:")
	Display_Matrix(b, false)
	fmt.Println("\nx:")
	Display_Matrix(x, false)

	// solve for x
	x_solved = Gauss_Jordan_Elimination(A, b, false)
	fmt.Println("x_solved by Gauss_Jordan_Elimination() unbatched:")
	Display_Matrix(x_solved, false)

	fmt.Println("Testing Gauss_Jordan_Elimination() batched")
	A = Zero_Tensor([]int{3, 3, 3}, true)
	A.Data = []float64{2, 4, -2, 4, 9, -3, -2, -3, 7, 2, 4, -2, 4, 9, -3, -2, -3, 7, 2, 4, -2, 4, 9, -3, -2, -3, 7}
	x = Zero_Tensor([]int{3, 3, 1}, true)
	x.Data = []float64{-1, 2, 2, -1, 2, 2, -1, 2, 2}
	b = MatMul(A, x, true)

	fmt.Println("\nA:")
	Display_Matrix(A, true)
	fmt.Println("\nb:")
	Display_Matrix(b, true)
	fmt.Println("\nx:")
	Display_Matrix(x, true)

	// solve for x
	x_solved = Gauss_Jordan_Elimination(A, b, true)
	fmt.Println("x_solved by Gauss_Jordan_Elimination() abtched:")
	Display_Matrix(x_solved, true)

	//-------------------------------------------------------------------------------------------------------------- AI_LinSys_Approximator()
}
