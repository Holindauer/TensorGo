package TG

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func Test_Elimination(t *testing.T) {

	fmt.Println("\n\nNow Testing Functions from matrix.go\n-------------------------------------")

	//-------------------------------------------------------------------------------------------------------------- Gaussian_Elimination()

	fmt.Print("Testing Gaussian_Elimination() unbatched...")

	// creat an A and x, multiply them to get b --- the solution will be x = [-1, 2, 2]
	A := Zero_Tensor([]int{3, 3}, false)
	A.Data = []float64{2, 4, -2, 4, 9, -3, -2, -3, 7}
	x := Zero_Tensor([]int{3, 1}, false)
	x.Data = []float64{-1, 2, 2}
	b := MatMul(A, x, false)

	// solve for x
	x_solved := Gaussian_Elimination(A, b, false)

	// check that the solution is correct (within 1e-10)
	if x_solved.Data[0] < -1.0000000001 || x_solved.Data[0] > -0.9999999999 {
		t.Error("Gaussian_Elimination() failed")
	}
	if x_solved.Data[1] < 1.9999999999 || x_solved.Data[1] > 2.0000000001 {
		t.Error("Gaussian_Elimination() failed")
	}
	if x_solved.Data[2] < 1.9999999999 || x_solved.Data[2] > 2.0000000001 {
		t.Error("Gaussian_Elimination() failed")
	}

	fmt.Println("Success!")

	fmt.Println("Testing Gaussian_Elimination() batched...")
	A = Zero_Tensor([]int{3, 3, 3}, true)
	A.Data = []float64{2, 4, -2, 4, 9, -3, -2, -3, 7, 2, 4, -2, 4, 9, -3, -2, -3, 7, 2, 4, -2, 4, 9, -3, -2, -3, 7}
	x = Zero_Tensor([]int{3, 3, 1}, true)
	x.Data = []float64{-1, 2, 2, -1, 2, 2, -1, 2, 2}
	b = MatMul(A, x, true)

	// solve for x
	x_solved = Gaussian_Elimination(A, b, true)

	// check that the solution is correct (within 1e-10)
	if x_solved.Data[0]+x_solved.Data[3]+x_solved.Data[6] < -3.0000000001 || x_solved.Data[0]+x_solved.Data[3]+x_solved.Data[6] > -2.9999999999 {
		t.Error("Gaussian_Elimination() failed")
	}
	if x_solved.Data[1]+x_solved.Data[4]+x_solved.Data[7] < 5.9999999999 || x_solved.Data[1]+x_solved.Data[4]+x_solved.Data[7] > 6.0000000001 {
		t.Error("Gaussian_Elimination() failed")
	}
	if x_solved.Data[2]+x_solved.Data[5]+x_solved.Data[8] < 5.9999999999 || x_solved.Data[2]+x_solved.Data[5]+x_solved.Data[8] > 6.0000000001 {
		t.Error("Gaussian_Elimination() failed")
	}

	//-------------------------------------------------------------------------------------------------------------- Gauss_Jordan_Elimination()

	fmt.Println("Testing Gauss_Jordan_Elimination() unbatched...")

	// creat an A and x, multiply them to get b
	A = Zero_Tensor([]int{3, 3}, false)
	A.Data = []float64{2, 4, -2, 4, 9, -3, -2, -3, 7}
	x = Zero_Tensor([]int{3, 1}, false)
	x.Data = []float64{-1, 2, 2}
	b = MatMul(A, x, false)

	// solve for x
	x_solved = Gauss_Jordan_Elimination(A, b, false)

	// check that the solution is correct (within 1e-10)
	if x_solved.Data[0] < -1.0000000001 || x_solved.Data[0] > -0.9999999999 {
		t.Error("Gauss_Jordan_Elimination() failed")
	}
	if x_solved.Data[1] < 1.9999999999 || x_solved.Data[1] > 2.0000000001 {
		t.Error("Gauss_Jordan_Elimination() failed")
	}
	if x_solved.Data[2] < 1.9999999999 || x_solved.Data[2] > 2.0000000001 {
		t.Error("Gauss_Jordan_Elimination() failed")
	}

	fmt.Println("Success!")

	fmt.Println("Testing Gauss_Jordan_Elimination() batched...")
	A = Zero_Tensor([]int{3, 3, 3}, true)
	A.Data = []float64{2, 4, -2, 4, 9, -3, -2, -3, 7, 2, 4, -2, 4, 9, -3, -2, -3, 7, 2, 4, -2, 4, 9, -3, -2, -3, 7}
	x = Zero_Tensor([]int{3, 3, 1}, true)
	x.Data = []float64{-1, 2, 2, -1, 2, 2, -1, 2, 2}
	b = MatMul(A, x, true)

	// solve for x
	x_solved = Gauss_Jordan_Elimination(A, b, true)

	// check that the solution is correct (within 1e-10)
	if x_solved.Data[0]+x_solved.Data[3]+x_solved.Data[6] < -3.0000000001 || x_solved.Data[0]+x_solved.Data[3]+x_solved.Data[6] > -2.9999999999 {
		t.Error("Gauss_Jordan_Elimination() failed")
	}
	if x_solved.Data[1]+x_solved.Data[4]+x_solved.Data[7] < 5.9999999999 || x_solved.Data[1]+x_solved.Data[4]+x_solved.Data[7] > 6.0000000001 {
		t.Error("Gauss_Jordan_Elimination() failed")
	}
	if x_solved.Data[2]+x_solved.Data[5]+x_solved.Data[8] < 5.9999999999 || x_solved.Data[2]+x_solved.Data[5]+x_solved.Data[8] > 6.0000000001 {
		t.Error("Gauss_Jordan_Elimination() failed")
	}

}
