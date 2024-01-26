package TG

// all_element_ops_test.go contains tests for all_element_ops.go

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

func Test_Sum_All(t *testing.T) {

	// @notice Testing Sum_All() Unbatched
	A := RangeTensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

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

	A := RangeTensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

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

func Test_Sum_Axis(t *testing.T) {

	// @notice Testing Sum_Axis() Unbatched

	// batch of individual 2D range tensor
	A := RangeTensor([]int{3, 3, 3}, true)

	// Set Batching flag to false
	A.Batched = false

	// Sum along axis 0
	summed := A.Sum_Axis(0, false)

	// Check that the sum is correct
	if summed.Sum_All() != 108 {
		t.Errorf("Sum_Axis() failed. Expected Output: 108 --- Actual Output: %v", summed.Sum_All())
	}

	// @notice Testing Sum_Axis() Batched

	// Add a singleton dim to the beginning of A
	A.Shape = []int{1, 3, 3, 3}

	// Concatenate A with itself twice
	A = A.Concat(A.Concat(A, 0), 0)

	// Set Batching flag to true
	A.Batched = true

	// Sum along axis 0
	summed = A.Sum_Axis(0, true)

	// Check that the sum is correct
	if summed.Sum_All() != 324 {
		t.Errorf("Sum_Axis() failed. Expected Output: 324 --- Actual Output: %v", summed.Sum_All())
	}
}

func Test_Mean_Axis(t *testing.T) {

	// @notice Testing Mean_Axis() Unbatched

	// <--- batch of individual 2D range tensor
	A := RangeTensor([]int{3, 3, 3}, true)

	// Set Batching flag to false
	A.Batched = false

	// Mean along axis 0
	meaned := A.Mean_Axis(0, false)

	// Check that the mean is correct
	if meaned.Sum_All() != 36 {
		t.Errorf("Mean_Axis() failed. Expected Output: 36 --- Actual Output: %v", meaned.Sum_All())
	}

	// @notice Testing Mean_Axis() Batched

	// Add a singleton dim to the beginning of A
	A.Shape = []int{1, 3, 3, 3}

	// Concatenate A with itself twice
	A = A.Concat(A.Concat(A, 0), 0)

	// Set Batching flag to true
	A.Batched = true

	// Mean along axis 0
	meaned = A.Mean_Axis(0, true)

	// Check that the mean is correct
	if meaned.Sum_All() != 108 {
		t.Errorf("Mean_Axis() failed. Expected Output: 108 --- Actual Output: %v", meaned.Sum_All())
	}
}

func Test_Var_Axis(t *testing.T) {

	// @notice Testing Var_Axis() Unbatched

	//  batch of individual 2D range tensor
	A := RangeTensor([]int{3, 3, 3}, true)

	// Set Batching flag to false
	A.Batched = false

	// Var along axis 0
	varianced := A.Var_Axis(0, false)

	// should be zero, all data points are the same along axis 0
	if varianced.Sum_All() != 0 {
		t.Errorf("Var_Axis() failed. Expected Output: 0 --- Actual Output: %v", varianced.Sum_All())
	}

	// @notice Testing Var_Axis() Batched

	// Add a singleton dim to the beginning of A
	A.Shape = []int{1, 3, 3, 3}

	// Concatenate A with itself twice
	A = A.Concat(A.Concat(A, 0), 0)

	// Set Batching flag to true
	A.Batched = true

	// Var along axis 0
	varianced = A.Var_Axis(0, true)

	// should be zero, all data points are the same along axis 0
	if varianced.Sum_All() != 0 {
		t.Errorf("Var_Axis() failed. Expected Output: 0 --- Actual Output: %v", varianced.Sum_All())
	}
}

func Test_Std_Axis(t *testing.T) {

	// @notice Testing Std_Axis() Unbatched

	// batch of individual 2D range tensor
	A := RangeTensor([]int{3, 3, 3}, true)

	// Set Batching flag to false
	A.Batched = false

	// Set Batching flag to false
	stded := A.Std_Axis(0, false) // <--- should be zero, all data points are the same along axis 0

	// should be zero, all data points are the same along axis 0
	if stded.Sum_All() != 0 {
		t.Errorf("Std_Axis() failed. Expected Output: 0 --- Actual Output: %v", stded.Sum_All())
	}

	// @notice Testing Std_Axis() Batched

	// Add a singleton dim to the beginning of A
	A.Shape = []int{1, 3, 3, 3}

	// Concatenate A with itself twice
	A = A.Concat(A.Concat(A, 0), 0)

	// Set Batching flag to true
	A.Batched = true

	// Take the std along axis 0
	stded = A.Std_Axis(0, true)

	// should be zero, all data points are the same along axis 0
	if stded.Sum_All() != 0 {
		t.Errorf("Std_Axis() failed. Expected Output: 0 --- Actual Output: %v", stded.Sum_All())
	}
}
