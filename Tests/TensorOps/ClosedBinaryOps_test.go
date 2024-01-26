package TG

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

func Test_Add(t *testing.T) {

	// @notice Testing Add() Unbatched

	// batch of individual 2D range tensor
	A := RangeTensor([]int{3, 3, 3}, true)

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

	A := RangeTensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

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
