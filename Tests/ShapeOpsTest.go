package TG

/*
* @notice ShapeOpsTest.go contains tests for functions in shape.go
 */

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

func Test_Slice(t *testing.T) {

	// Create a 3x3x3 tensor
	A := Range_Tensor([]int{3, 3, 3}, true)

	// Creates a 3x2x2 tensor
	A_partial := A.Slice(":3, 1:3, 1:3")
	//Display_Matrix(A_partial, true)
	A_partial_extracted := A_partial.GetBatchElement(0)

	// Check that the sum of the elements is 24 and that the shape is 2x2
	if A_partial_extracted.Sum_All() != 24 && len(A_partial_extracted.Shape) != 2 {
		t.Errorf("Slice() failed. Expected Output: 24 --- Actual Output: %v", A_partial_extracted.Sum_All())
	}
}

func Test_Reshape(t *testing.T) {

	/// @noice Test Reshape() Unbatched
	A := Range_Tensor([]int{3, 3, 3}, false)
	A_reshaped := A.Reshape([]int{3, 9}, false)

	// Ensure data has not changed, just shape
	if A.Sum_All() != A_reshaped.Sum_All() {
		t.Errorf("Reshape() failed. Expected Output: %v --- Actual Output: %v", A.Sum_All(), A_reshaped.Sum_All())
	}

	/// @noice Test Reshape() Batched
	A = Range_Tensor([]int{3, 3, 3}, true)

	// Reshape each 3x3 element into a 9D vector
	A_reshaped = A.Reshape([]int{9}, true)

	// Extract the first element of the batch of the og and reshaped tensors
	A_reshaped_extracted := A_reshaped.GetBatchElement(0)
	A_extracted := A.GetBatchElement(0)

	// Ensure data has not changed, just shape
	if A_extracted.Sum_All() != A_reshaped_extracted.Sum_All() {
		t.Errorf("Reshape() failed. Expected Output: %v --- Actual Output: %v", A_extracted.Sum_All(), A_reshaped_extracted.Sum_All())
	}
}

func Test_Permute(t *testing.T) {

	// Test Permute()

	// Create a 3x3x6x8x9 tensor
	A := Range_Tensor([]int{3, 3, 6, 8, 9}, true)

	// Permute the tensor dimmensions
	A_permuted := A.Permute([]int{1, 0, 3, 2, 4})

	// Check that a pemuted index is equal to the original index
	if A.Get([]int{0, 1, 2, 3, 4}) != A_permuted.Get([]int{1, 0, 3, 2, 4}) {
		t.Errorf("Permute() failed. Same element after transpose not equal. Expected Output: %v --- Actual Output: %v", A.Get([]int{0, 1, 2, 3, 4}), A_permuted.Get([]int{1, 0, 3, 2, 4}))
	}

}

func Test_Concat(t *testing.T) {
	// Test Concat()

	// Create two 3x3x3 tensors
	A := Range_Tensor([]int{1, 3, 3}, true)
	B := Range_Tensor([]int{1, 3, 3}, true)

	// Concatenate the two tensors along the first axis
	C := A.Concat(B, 0)

	// Extract the first element of the batch of the og and reshaped tensors
	A_extracted := A.GetBatchElement(0)
	B_extracted := B.GetBatchElement(0)

	// Ensure the extracted elements sum to the same value as the concatenated tensor
	if A_extracted.Sum_All()+B_extracted.Sum_All() != C.Sum_All() {
		t.Errorf("Concat() failed. Expected Output: %v --- Actual Output: %v", A_extracted.Sum_All()+B_extracted.Sum_All(), C.Sum_All())
	}
}

func Test_ExtendShape(t *testing.T) {
	// Test Extend_Shape()

	// Create a 3x3x3 Range tensor
	A := Range_Tensor([]int{3, 3, 3}, true)

	// Extend the shape of A to 4 dimensions
	A_extended := A.Extend_Shape(4)

	// Ensure there is an extra dimension and that the sum of the elements is 4x the original sum
	if len(A_extended.Shape) != 4 {
		t.Errorf("Extend_Shape() failed. Expected Output: 4 --- Actual Output: %v", len(A_extended.Shape))
	}

	// Ensure the sum of the elements is 4x the original sum
	if A_extended.Sum_All() != A.Sum_All()*4 {
		t.Errorf("Extend_Shape() failed. Expected Output: %v --- Actual Output: %v", A.Sum_All(), A_extended.Sum_All())
	}
}

func Test_ExtendDim(t *testing.T) {

	// Create a 3x3x3 Range tensor
	A := Range_Tensor([]int{3, 3, 3}, true)

	// Extend the second axis by 3 elements
	A_extended := A.Extend_Dim(2, 3)

	// Ensure the third axis is now 6 elements long and that the sum of the elements is the same
	if A_extended.Shape[2] != 6 {
		t.Errorf("Extend_Dim() failed. Expected Output: 6 --- Actual Output: %v", A_extended.Shape[2])
	}

	// Ensure the sum of the elements is the same (because the new elements are all 0)
	if A_extended.Sum_All() != A.Sum_All() {
		t.Errorf("Extend_Dim() failed. Expected Output: %v --- Actual Output: %v", A.Sum_All(), A_extended.Sum_All())
	}
}

func Test_Remove_Dim(t *testing.T) {

	// Test Remove_Dim()

	// Init a 3x3x3 range tensor
	A := Range_Tensor([]int{3, 3, 3}, true)

	// Remove the first axis, keeping the first element of the first axis
	A_removed := A.Remove_Dim(0, 0)

	// Ensure the shape is now 2x3x3 and that the sum of the elements is 1/3 of the original sum
	if len(A_removed.Shape) != 2 {
		t.Errorf("Remove_Dim() failed. Expected Output: 2 --- Actual Output: %v", len(A_removed.Shape))
	}

	// Ensure the sum of the elements is 1/3 of the original sum
	if A_removed.Sum_All() != A.Sum_All()/3 {
		t.Errorf("Remove_Dim() failed. Expected Output: %v --- Actual Output: %v", A.Sum_All(), A_removed.Sum_All())
	}

}

func Test_Add_Singleton(t *testing.T) {

	// Test Add_Singleton()

	// Init a 3x3x3 range tensor
	A := Range_Tensor([]int{3, 3, 3}, true)

	// Add a singleton dimension to the first axis
	A_added := A.Add_Singleton(1)

	// Ensure we have added a singleton dimension
	if len(A_added.Shape) != 4 {
		t.Errorf("Add_Singleton() failed. Expected Output: 4 --- Actual Output: %v", len(A_added.Shape))
	}

}

func Test_Remove_Singleton(t *testing.T) {
	// Test Remove_Singleton()

	// Init a 3x1x1x3x3x1 range tensor
	A := Range_Tensor([]int{3, 1, 1, 3, 3, 1}, true)

	// Remove all singleton dimensions
	A_removed := A.Remove_Singletons()

	// Ensure we have removed all singleton dimensions
	if len(A_removed.Shape) != 3 {
		t.Errorf("Remove_Singleton() failed. Expected Output: 3 --- Actual Output: %v", len(A_removed.Shape))
	}
}
