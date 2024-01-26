package TG

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

func Test_Normalize(t *testing.T) {

	// @notice Testing Normalize() Unbatched

	//  batch of individual 2D range tensor
	A := RangeTensor([]int{3, 3, 3}, true)

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
