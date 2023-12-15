package TG

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func Test_Axis_Inplace_Operations(t *testing.T) {

	fmt.Println("\n\nNow Testing Functions from axis_inplace_ops.go\n-------------------------------------------------")

	// Testing Add()

	fmt.Print("Testing Add() Unbatched...")

	A := Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	Added := Add(A, A, false)

	if Added.Sum_All() != 216 {
		t.Errorf("Add() failed. Expected Output: 216 --- Actual Output: %v", Added.Sum_All())
	}

	fmt.Println("Success!")

	fmt.Print("Testing Add() Batched...")

	A.Shape = []int{1, 3, 3, 3}
	A = A.Concat(A.Concat(A, 0), 0)

	Added = Add(A, A, true)

	if Added.Sum_All() != 648 {
		t.Errorf("Add() failed. Expected Output: 648 --- Actual Output: %v", Added.Sum_All())
	}

	fmt.Println("Success!")

	// Testing Subtract()

	fmt.Print("Testing Subtract() Unbatched...")

	A = Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	Subtracted := Subtract(A, A, false)

	if Subtracted.Sum_All() != 0 {
		t.Errorf("Subtract() failed. Expected Output: 0 --- Actual Output: %v", Subtracted.Sum_All())
	}

	fmt.Println("Success!")

	fmt.Print("Testing Subtract() Batched...")
	A.Shape = []int{1, 3, 3, 3}

	A = A.Concat(A.Concat(A, 0), 0)

	Subtracted = Subtract(A, A, true)

	if Subtracted.Sum_All() != 0 {
		t.Errorf("Subtract() failed. Expected Output: 0 --- Actual Output: %v", Subtracted.Sum_All())
	}

	fmt.Println("Success!")

	// Testing Scalar_Mult_()

	fmt.Print("Testing Scalar_Mult_() Unbatched...")

	A = Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	Scalar_Multed := A.Scalar_Mult(2, false)

	if Scalar_Multed.Sum_All() != 216 {

		t.Errorf("Scalar_Mult_() failed. Expected Output: 216 --- Actual Output: %v", Scalar_Multed.Sum_All())
	}

	fmt.Println("Success!")

	fmt.Print("Testing Scalar_Mult_() Batched...")

	A.Shape = []int{1, 3, 3, 3}

	A = A.Concat(A.Concat(A, 0), 0)

	Scalar_Multed = A.Scalar_Mult(2, true)

	if Scalar_Multed.Sum_All() != 648 {

		t.Errorf("Scalar_Mult_() failed. Expected Output: 648 --- Actual Output: %v", Scalar_Multed.Sum_All())
	}

	fmt.Println("Success!")

	// Testing Normalize()

	fmt.Print("Testing Normalize() Unbatched...")

	A = Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	Normalized := A.Normalize(false)

	for i := 0; i < len(Normalized.Data); i++ {
		if !(Normalized.Data[i] < 1) {
			t.Errorf("Normalize() failed. Expected Output: 1 --- Actual Output: %v", Normalized.Data[i])
		}
	}

	fmt.Println("Success!")

	fmt.Print("Testing Normalize() Batched...")

	A.Shape = []int{1, 3, 3, 3}

	A = A.Concat(A.Concat(A, 0), 0)

	Normalized = A.Normalize(true)

	for i := 0; i < len(Normalized.Data); i++ {
		if !(Normalized.Data[i] < 1) {
			t.Errorf("Normalize() failed. Expected Output: 1 --- Actual Output: %v", Normalized.Data[i])
		}
	}

	fmt.Println("Success!")
}
