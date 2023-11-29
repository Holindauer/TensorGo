package GLA

// This source file contains tests of operations applied along an axis that retain the Tensor shape

import (
	"fmt"

	"testing"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func Test_Tensor_Axis_InplaceOps(t *testing.T) {

	//-------------------------------------------------------------------------------------------------------------- Normalize_Axis()

	fmt.Println("\nTesting Normalize_Axis() unbatched...") // currently no batched version of this function

	// Test Normalize_Axis()
	A := Range_Tensor([]int{3, 4}, false)

	fmt.Println("A:")
	Display_Matrix(A, false)

	fmt.Println("A_Normalized:")
	A_Normalized := A.Normalize_Axis(0) // normalize along the first axis

	Display_Matrix(A_Normalized, false)
}
