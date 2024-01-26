package TG

/*
* @notice broadcast_test.go contains tests for functions in broadcast.go
 */
import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

func Test_BroadcastAdd(t *testing.T) {

	// Testing Broadcast_Add() by adding a 3x3x3 batched ones tensor to a 3x3 unbached ones tensor
	A := OnesTensor([]int{3, 3, 3}, true)
	B := OnesTensor([]int{3, 3}, false)

	// Broadcast_Addition pf A onto B
	B_broad_A := B.Broadcast_Add(A)

	if B_broad_A.Sum_All() != 54 {
		t.Errorf("Broadcast_Add() failed. Expected Output: 36 --- Actual Output: %v", B_broad_A.Sum_All())
	}
}

func Test_BroadcastSubtract(t *testing.T) {
	// Testing Broadcast_Subtract() by subtracting a 3x3x3 batched ones tensor to a 3x3 unbached ones tensor
	A := OnesTensor([]int{3, 3, 3}, true)
	B := OnesTensor([]int{3, 3}, false)

	// Broadcast_Addition pf A onto B
	B_broad_A := B.Broadcast_Subtract(A)

	if B_broad_A.Sum_All() != 0 {
		t.Errorf("Broadcast_Subtract() failed. Expected Output: 0 --- Actual Output: %v", B_broad_A.Sum_All())
	}

}

func Test_Scalar_Mult(t *testing.T) {

	// @ notice Testing Scalar_Mult_() Unbatched

	// batch of individual 2D range tensor
	A := RangeTensor([]int{3, 3, 3}, true)

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
