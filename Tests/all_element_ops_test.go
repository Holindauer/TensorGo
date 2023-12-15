package TG

// all_element_ops_test.go contains tests for all_element_ops.go

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func Test_All_Element_Operations(t *testing.T) {

	fmt.Println("\n\nNow Testing Functions from all_element_ops.go\n-------------------------------------------------")

	// Testing Sum_All()

	fmt.Print("Testing Sum_All() Unbatched...")

	A := Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	if A.Sum_All() != 108 {
		t.Errorf("Sum_All() failed. Expected Output: 108 --- Actual Output: %v", A.Sum_All())
	}

	fmt.Println("Success!")

	fmt.Print("Testing Sum_All() Batched...")

	A.Shape = []int{1, 3, 3, 3}
	A = A.Concat(A.Concat(A, 0), 0)

	if A.Sum_All() != 324 {
		t.Errorf("Sum_All() failed. Expected Output: 324 --- Actual Output: %v", A.Sum_All())
	}

	fmt.Println("Success!")

	// Testing Mean_All()

	fmt.Print("Testing Mean_All() Unbatched...")

	A = Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	// Test the mean within a tolerance of (1e-10)
	if A.Mean_All() < 3.9999999999 || A.Mean_All() > 4.0000000001 {

		t.Errorf("Mean_All() failed. Expected Output: 4 --- Actual Output: %v", A.Mean_All())
	}

	fmt.Println("Success!")

	fmt.Print("Testing Mean_All() Batched...")

	A.Shape = []int{1, 3, 3, 3}

	A = A.Concat(A.Concat(A, 0), 0)

	if A.Mean_All() < 3.9 || A.Mean_All() > 4.1 {

		t.Errorf("Mean_All() failed. Expected Output: 4 --- Actual Output: %v", A.Mean_All())
	}

	fmt.Println("Success!")

	// Determine a way to TestVar_All(), and Std_All()
}
