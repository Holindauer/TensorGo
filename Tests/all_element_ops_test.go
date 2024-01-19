package TG

// all_element_ops_test.go contains tests for all_element_ops.go

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

func Test_Sum_All(t *testing.T) {

	// @notice Testing Sum_All() Unbatched
	A := Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	// set Batching flag to false
	A.Batched = false

	// Test is the expected output is equal to the actual output
	if A.Sum_All() != 108 {
		t.Errorf("Sum_All() Unbatched failed. Expected Output: 108 --- Actual Output: %v", A.Sum_All())
	}

	// @notice Testing Sum_All() Batched

	// Add a singleton dim to the beginning of A
	A.Shape = []int{1, 3, 3, 3}

	// Concatenate A with itself twice
	A = A.Concat(A.Concat(A, 0), 0)

	// Set Batching flag to true
	A.Batched = true

	// Test is the expected output is equal to the actual output
	if A.Sum_All() != 324 {
		t.Errorf("Sum_All() Batched failed. Expected Output: 324 --- Actual Output: %v", A.Sum_All())
	}
}

func Test_Mean_All(t *testing.T) {

	// @notice Testing Mean_All() Unbatched

	A := Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	// Test the mean within a tolerance of (1e-10)
	if A.Mean_All() < 3.9999999999 || A.Mean_All() > 4.0000000001 {
		t.Errorf("Mean_All() failed. Expected Output: 4 --- Actual Output: %v", A.Mean_All())
	}

	// @notice Testing Mean_All() Batched

	// Add a singleton dim to the beginning of A
	A.Shape = []int{1, 3, 3, 3}

	// Concatenate A with itself twice
	A = A.Concat(A.Concat(A, 0), 0)

	// Test the mean within a tolerance of (1e-10)
	if A.Mean_All() < 3.9 || A.Mean_All() > 4.1 {
		t.Errorf("Mean_All() failed. Expected Output: 4 --- Actual Output: %v", A.Mean_All())
	}
}

// TODO --- Testing still needs to be implemented for Var_All and Std_All
