package TG

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

func Test_Add(t *testing.T) {

	// @notice Testing Add() Unbatched

	// batch of individual 2D range tensor
	A := Range_Tensor([]int{3, 3, 3}, true)

	// Se Batching flag to false
	A.Batched = false

	// Elementwise Add A with itself unbatched
	Added := Add(A, A, false)

	if Added.Sum_All() != 216 {
		t.Errorf("Add() Unbatched failed. Expected Output: 216 --- Actual Output: %v", Added.Sum_All())
	}

	// @notice Testing Add() Batched

	// Add a singleton dim to the beginning of A
	A.Shape = []int{1, 3, 3, 3}

	// Concatenate A with itself twice
	A = A.Concat(A.Concat(A, 0), 0)

	// Set Batching flag to true
	A.Batched = true

	// Elementwise add A with itself batched
	Added = Add(A, A, true)

	if Added.Sum_All() != 648 {
		t.Errorf("Add() Unbatched failed. Expected Output: 648 --- Actual Output: %v", Added.Sum_All())
	}
}

func Test_Subtract(t *testing.T) {

	// @notice Testing Subtract() Unbatched

	A := Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	// Set Batching flag to false
	A.Batched = false

	// Elementwise Subtract A from itself unbatched
	Subtracted := Subtract(A, A, false)

	if Subtracted.Sum_All() != 0 {
		t.Errorf("Subtract() Unbatched Failed. Expected Output: 0 --- Actual Output: %v", Subtracted.Sum_All())
	}

	// @notice Testing Subtract() Batched

	// Add a singleton dim to the beginning of A
	A.Shape = []int{1, 3, 3, 3}

	// Concatenate A with itself twice
	A = A.Concat(A.Concat(A, 0), 0)

	// Set Batching flag to true
	A.Batched = true

	// Elementwise Subtract A from itself batched
	Subtracted = Subtract(A, A, true)

	if Subtracted.Sum_All() != 0 {
		t.Errorf("Subtract() Batched Failed. Expected Output: 0 --- Actual Output: %v", Subtracted.Sum_All())
	}
}

func Test_Scalar_Mult(t *testing.T) {

	// @ notice Testing Scalar_Mult_() Unbatched

	// batch of individual 2D range tensor
	A := Range_Tensor([]int{3, 3, 3}, true)

	// Set Batching flag to false
	A.Batched = false

	// Scalar_Mult A by 2 unbatched
	Scalar_Multed := A.Scalar_Mult(2, false)

	// Test is the expected output is equal to the actual output
	if Scalar_Multed.Sum_All() != 216 {

		t.Errorf("Scalar_Mult_() Unbatched failed. Expected Output: 216 --- Actual Output: %v", Scalar_Multed.Sum_All())
	}

	// @notice Testing Scalar_Mult_() Batched

	// Add a singleton dim to the beginning of A
	A.Shape = []int{1, 3, 3, 3}

	// Concatenate A with itself twice
	A = A.Concat(A.Concat(A, 0), 0)

	// Set Batching flag to true
	Scalar_Multed = A.Scalar_Mult(2, true)

	if Scalar_Multed.Sum_All() != 648 {

		t.Errorf("Scalar_Mult_() Batched failed. Expected Output: 648 --- Actual Output: %v", Scalar_Multed.Sum_All())
	}
}

func Test_Normalize(t *testing.T) {

	// @notice Testing Normalize() Unbatched

	//  batch of individual 2D range tensor
	A := Range_Tensor([]int{3, 3, 3}, true)

	// Set Batching flag to false
	A.Batched = false

	// Normalize A unbatched
	Normalized := A.Normalize(false)

	// Test whether the output is normalized to (0, 1)
	for i := 0; i < len(Normalized.Data); i++ {
		if !(Normalized.Data[i] < 1) {
			t.Errorf("Normalize() failed. Expected Output: 1 --- Actual Output: %v", Normalized.Data[i])
		}
	}

	// @notice Testing Normalize() Batched

	// Add a singleton dim to the beginning of A
	A.Shape = []int{1, 3, 3, 3}

	// Concatenate A with itself twice
	A = A.Concat(A.Concat(A, 0), 0)

	// Set Batching flag to true
	A.Batched = true

	// Normalize A batched
	Normalized = A.Normalize(true)

	// Test whether the output is normalized to (0, 1)
	for i := 0; i < len(Normalized.Data); i++ {
		if !(Normalized.Data[i] < 1) {
			t.Errorf("Normalize() failed. Expected Output: 1 --- Actual Output: %v", Normalized.Data[i])
		}
	}
}
