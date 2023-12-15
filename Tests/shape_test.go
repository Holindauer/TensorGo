package TG

// shape_test.go contains tests for functions in shape.go

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func Test_Shape(t *testing.T) {

	fmt.Println("\n\nNow Testing Functions from shape.go\n-------------------------------------")

	// Test Partial()
	fmt.Print("Testing Partial()...")

	A := Range_Tensor([]int{3, 3, 3}, true)

	// Creates a 3x2x2 tensor
	A_partial := A.Partial(":3, 1:3, 1:3")
	//Display_Matrix(A_partial, true)
	A_partial_extracted := A_partial.Extract(0)

	if A_partial_extracted.Sum_All() != 24 && len(A_partial_extracted.Shape) != 2 {
		t.Errorf("Partial() failed. Expected Output: 24 --- Actual Output: %v", A_partial_extracted.Sum_All())
	}

	fmt.Println("Succsess!")

	// Test Reshape()

	fmt.Print("Testing Reshape() Unbatched...")
	A = Range_Tensor([]int{3, 3, 3}, false)
	A_reshaped := A.Reshape([]int{3, 9}, false)

	if A.Sum_All() != A_reshaped.Sum_All() {
		t.Errorf("Reshape() failed. Expected Output: %v --- Actual Output: %v", A.Sum_All(), A_reshaped.Sum_All())
	}

	fmt.Println("Succsess!")

	fmt.Print("Testing Reshape() Batched...")
	A = Range_Tensor([]int{3, 3, 3}, true)

	A_reshaped = A.Reshape([]int{9}, true)

	A_reshaped_extracted := A_reshaped.Extract(0)
	A_extracted := A.Extract(0)

	if A_extracted.Sum_All() != A_reshaped_extracted.Sum_All() {
		t.Errorf("Reshape() failed. Expected Output: %v --- Actual Output: %v", A_extracted.Sum_All(), A_reshaped_extracted.Sum_All())
	}

	fmt.Println("Succsess!")

	// Test Transpose()
	fmt.Print("Testing Transpose()...")

	A = Range_Tensor([]int{3, 3, 6, 8, 9}, true)
	A_transposed := A.Transpose([]int{1, 0, 3, 2, 4})

	if A.Retrieve([]int{0, 1, 2, 3, 4}) != A_transposed.Retrieve([]int{1, 0, 3, 2, 4}) {
		t.Errorf("Transpose() failed. Same element after transpose not equal. Expected Output: %v --- Actual Output: %v", A.Retrieve([]int{0, 1, 2, 3, 4}), A_transposed.Retrieve([]int{1, 0, 3, 2, 4}))
	}

	fmt.Println("Succsess!")

	// Test Concat()

	fmt.Print("Testing Concat()...")

	A = Range_Tensor([]int{1, 3, 3}, true)
	B := Range_Tensor([]int{1, 3, 3}, true)
	C := A.Concat(B, 0)

	A_extracted = A.Extract(0)
	B_extracted := B.Extract(0)

	if A_extracted.Sum_All()+B_extracted.Sum_All() != C.Sum_All() {
		t.Errorf("Concat() failed. Expected Output: %v --- Actual Output: %v", A_extracted.Sum_All()+B_extracted.Sum_All(), C.Sum_All())
	}

	fmt.Println("Succsess!")

	// Test Extend_Shape()

	fmt.Print("Testing Extend_Shape()...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	A_extended := A.Extend_Shape(4)

	if len(A_extended.Shape) != 4 {
		t.Errorf("Extend_Shape() failed. Expected Output: 4 --- Actual Output: %v", len(A_extended.Shape))
	}

	if A_extended.Sum_All() != A.Sum_All()*4 {
		t.Errorf("Extend_Shape() failed. Expected Output: %v --- Actual Output: %v", A.Sum_All(), A_extended.Sum_All())
	}

	fmt.Println("Succsess!")

	// Test Extend_Dim()

	fmt.Print("Testing Extend_Dim()...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	A_extended = A.Extend_Dim(2, 3)

	if A_extended.Shape[2] != 6 {
		t.Errorf("Extend_Dim() failed. Expected Output: 6 --- Actual Output: %v", A_extended.Shape[2])
	}

	if A_extended.Sum_All() != A.Sum_All() {
		t.Errorf("Extend_Dim() failed. Expected Output: %v --- Actual Output: %v", A.Sum_All(), A_extended.Sum_All())
	}

	fmt.Println("Succsess!")

	// Test Remove_Dim()

	fmt.Print("Testing Remove_Dim()...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	A_removed := A.Remove_Dim(0, 0)

	if len(A_removed.Shape) != 2 {
		t.Errorf("Remove_Dim() failed. Expected Output: 2 --- Actual Output: %v", len(A_removed.Shape))
	}

	if A_removed.Sum_All() != A.Sum_All()/3 {
		t.Errorf("Remove_Dim() failed. Expected Output: %v --- Actual Output: %v", A.Sum_All(), A_removed.Sum_All())
	}

	fmt.Println("Succsess!")

	// Test Add_Singleton()

	fmt.Print("Testing Add_Singleton()...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	A_added := A.Add_Singleton()

	if len(A_added.Shape) != 4 {
		t.Errorf("Add_Singleton() failed. Expected Output: 4 --- Actual Output: %v", len(A_added.Shape))
	}

	// Test Remove_Singleton()
	A = Range_Tensor([]int{3, 1, 1, 3, 3, 1}, true)
	A_removed = A.Remove_Singletons()

	if len(A_removed.Shape) != 3 {
		t.Errorf("Remove_Singleton() failed. Expected Output: 3 --- Actual Output: %v", len(A_removed.Shape))
	}

}
