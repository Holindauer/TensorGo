package GLA

// This source file contains tests of operations regarding matricies within the matrix.go source file.

import (
	"fmt"

	"testing"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func Test_Matrix(t *testing.T) {

	fmt.Println("Now Testing Matrix Operations:\n===============================")

	// Test unbatched matmul
	fmt.Println("Testing Unbatched Matmul:\n--------------------------")
	A := Range_Tensor([]int{3, 3}, false)
	B := Range_Tensor([]int{3, 3}, false)

	fmt.Println("\nA:")
	Display_Matrix(A, false)
	fmt.Println("\nB:")
	Display_Matrix(B, false)

	C := MatMul(A, B, false)

	fmt.Println("\nC:")
	Display_Matrix(C, false)

	// Test Batched Matmul
	fmt.Println("\nTesting Batched Matmul:\n------------------------")
	A = Range_Tensor([]int{2, 3, 3}, true)
	B = Range_Tensor([]int{2, 3, 3}, true)

	fmt.Println("\nA:")
	Display_Matrix(A, true)
	fmt.Println("\nB:")
	Display_Matrix(B, true)

	C = MatMul(A, B, true)

	fmt.Println("\nC:")
	Display_Matrix(C, true)

	// Test Augment_Matrix (currently no batching)
	fmt.Println("\nTesting Augment_Matrix:\n------------------------")
	A = Range_Tensor([]int{3, 3}, false)
	B = Range_Tensor([]int{3, 3}, false)

	fmt.Println("\nA Prior to Augmentation:")
	Display_Matrix(A, false)

	fmt.Println("\nB Prior to Augmentation:")
	Display_Matrix(B, false)

	C = Augment_Matrix(A, B)

	fmt.Println("\nC After Augmentation:")
	Display_Matrix(C, false)

}
