package TG

// axis_collapsing_ops_test.go contains tests for functions in axis_collapsing_ops.go

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func Test_Axis_Collapsing_Operations(t *testing.T) {

	fmt.Println("\n\nNow Testing Functions from axis_collapsing_ops.go\n-------------------------------------------------")

	// Testing Sum_Axis()
	fmt.Print("Testing Sum_Axis() Unbatched...")

	A := Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor
	summed := A.Sum_Axis(0, false)

	if summed.Sum_All() != 108 {
		t.Errorf("Sum_Axis() failed. Expected Output: 108 --- Actual Output: %v", summed.Sum_All())
	}
	fmt.Println("Success!")

	fmt.Print("Testing Sum_Axis() Batched...")
	A.Shape = []int{1, 3, 3, 3}
	A = A.Concat(A.Concat(A, 0), 0)

	summed = A.Sum_Axis(0, true)

	if summed.Sum_All() != 324 {
		t.Errorf("Sum_Axis() failed. Expected Output: 324 --- Actual Output: %v", summed.Sum_All())
	}

	fmt.Println("Success!")

	// Testing Mean_Axis()

	fmt.Print("Testing Mean_Axis() Unbatched...")
	A = Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	meaned := A.Mean_Axis(0, false)

	if meaned.Sum_All() != 36 {
		t.Errorf("Mean_Axis() failed. Expected Output: 36 --- Actual Output: %v", meaned.Sum_All())
	}

	fmt.Println("Success!")

	fmt.Print("Testing Mean_Axis() Batched...")
	A.Shape = []int{1, 3, 3, 3}
	A = A.Concat(A.Concat(A, 0), 0)

	meaned = A.Mean_Axis(0, true)

	if meaned.Sum_All() != 108 {
		t.Errorf("Mean_Axis() failed. Expected Output: 108 --- Actual Output: %v", meaned.Sum_All())
	}
	fmt.Println("Success!")

	// Testing Var_Axis()

	fmt.Print("Testing Var_Axis() Unbatched...")
	A = Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	varianced := A.Var_Axis(0, false) // <--- should be zero, all data points are the same along axis 0

	if varianced.Sum_All() != 0 {
		t.Errorf("Var_Axis() failed. Expected Output: 0 --- Actual Output: %v", varianced.Sum_All())
	}

	fmt.Println("Success!")

	fmt.Print("Testing Var_Axis() Batched...")
	A.Shape = []int{1, 3, 3, 3}
	A = A.Concat(A.Concat(A, 0), 0)

	varianced = A.Var_Axis(0, true)

	if varianced.Sum_All() != 0 {
		t.Errorf("Var_Axis() failed. Expected Output: 0 --- Actual Output: %v", varianced.Sum_All())
	}
	fmt.Println("Success!")

	// Testing Std_Axis()

	fmt.Print("Testing Std_Axis() Unbatched...")

	A = Range_Tensor([]int{3, 3, 3}, true) // <--- batch of individual 2D range tensor

	stded := A.Std_Axis(0, false) // <--- should be zero, all data points are the same along axis 0

	if stded.Sum_All() != 0 {
		t.Errorf("Std_Axis() failed. Expected Output: 0 --- Actual Output: %v", stded.Sum_All())
	}

	fmt.Println("Success!")

	fmt.Print("Testing Std_Axis() Batched...")
	A.Shape = []int{1, 3, 3, 3}
	A = A.Concat(A.Concat(A, 0), 0)

	stded = A.Std_Axis(0, true)

	if stded.Sum_All() != 0 {
		t.Errorf("Std_Axis() failed. Expected Output: 0 --- Actual Output: %v", stded.Sum_All())
	}
	fmt.Println("Success!")

}
