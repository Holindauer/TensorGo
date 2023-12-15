package TG

// tensor_test.go contains tests for functions in tensor.go

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func Test_Tensor(t *testing.T) {

	fmt.Println("\n\nNow Testing Functions from tensor.go\n-------------------------------------")

	// Test Index()
	fmt.Print("Testing Index()...")
	A := Range_Tensor([]int{10, 12, 14}, false)
	if A.Index([]int{2, 4, 3}) != 395 {
		t.Errorf("Index() failed. Expected Output: 395 --- Actual Output: %v", A.Index([]int{2, 4, 3}))
	}

	fmt.Println("Succsess!")

	// Test Retrieve()
	fmt.Print("Testing Retrieve()...")
	if A.Retrieve([]int{2, 4, 3}) != A.Data[395] {
		t.Errorf("Retrieve() failed. Expected Output: %v --- Actual Output: %v", A.Data[395], A.Retrieve([]int{2, 4, 3}))
	}

	fmt.Println("Succsess!")

	// Test UnravelIndex()
	fmt.Print("Testing UnravelIndex()...")
	Unraveled := A.UnravelIndex(395)
	if Unraveled[0] != 2 || Unraveled[1] != 4 || Unraveled[2] != 3 {
		t.Errorf("UnravelIndex() failed. Expected Output: [2, 4, 3] --- Actual Output: %v", Unraveled)
	}

	fmt.Println("Succsess!")

	// Test Extract()
	fmt.Print("Testing Extract()...")

	// Concatenate 2 range tensors, then 1 ones Tensor, then 2 range ones tensor
	ranges_1 := Range_Tensor([]int{2, 10, 10}, true)
	randoms := Ones_Tensor([]int{1, 10, 10}, true)
	ranges_2 := Range_Tensor([]int{2, 10, 10}, true)

	batched_tensors := ranges_1.Concat(randoms, 0)
	batched_tensors = batched_tensors.Concat(ranges_2, 0)

	fmt.Println("Succsess!")

	// Extract the random tensor
	fmt.Print("Testing Extract()...")
	accessed_element := batched_tensors.Extract(2)

	if accessed_element.Sum_All() != 100 {
		t.Errorf("Extract() failed. Expected a 10x10 ones Tensor --- Actual Output:")
		Display_Matrix(accessed_element, false)
	}

	fmt.Println("Succsess!")
}
