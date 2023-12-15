package TG

// matrix_test.go contains tests for the matrix.go file

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func Test_Matrix(t *testing.T) {

	fmt.Println("\n\nNow Testing Functions from matrix.go\n-------------------------------------")

	// Test MatMul()

	fmt.Print("Testing MatMul() Unbatched...")
	A := Range_Tensor([]int{3, 3}, false)
	B := Range_Tensor([]int{3, 3}, false)

	C := MatMul(A, B, false)

	if C.Sum_All() != 486 {
		t.Error("MatMul() failed")
	}

	fmt.Println("Success!")

	fmt.Print("Testing MatMul() Batched...")

	A = Range_Tensor([]int{3, 3, 3}, true)
	B = Range_Tensor([]int{3, 3, 3}, true)

	C = MatMul(A, B, true)

	if C.Sum_All() != 486*3 {
		t.Error("MatMul() failed")
	}

	fmt.Println("Success!")

	// Augment Matrix

	fmt.Print("Testing Augment_Matrix()...")

	A = Range_Tensor([]int{3, 3}, false)
	B = Range_Tensor([]int{3, 3}, false)

	C = Augment_Matrix(A, B)

	if C.Shape[0] != 3 || C.Shape[1] != 6 {
		t.Error("Augment_Matrix() failed")
	}
	if C.Sum_All() != A.Sum_All()+B.Sum_All() {
		t.Error("Augment_Matrix() failed")
	}
}
