package TG

// matrix_test.go contains tests for the matrix.go file

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

func Test_MatMul(t *testing.T) {

	/// @notice Test MatMul() Unbatched
	A := Range_Tensor([]int{3, 3}, false)
	B := Range_Tensor([]int{3, 3}, false)

	C := MatMul(A, B, false)
	if C.Sum_All() != 486 {
		t.Error("MatMul() failed")
	}

	/// @notice Test MatMul() Batched
	A = Range_Tensor([]int{3, 3, 3}, true)
	B = Range_Tensor([]int{3, 3, 3}, true)

	C = MatMul(A, B, true)

	if C.Sum_All() != 486*3 {
		t.Error("MatMul() failed")
	}
}

func Test_Augment_Matrix(t *testing.T) {

	// Test Augment Matrix
	A := Range_Tensor([]int{3, 3}, false)
	B := Range_Tensor([]int{3, 3}, false)

	// Augment A with B
	C := Augment_Matrix(A, B)

	// Check for correct shape
	if C.Shape[0] != 3 || C.Shape[1] != 6 {
		t.Error("Augment_Matrix() failed")
	}

	// Check for correct sum of values
	if C.Sum_All() != A.Sum_All()+B.Sum_All() {
		t.Error("Augment_Matrix() failed")
	}
}
