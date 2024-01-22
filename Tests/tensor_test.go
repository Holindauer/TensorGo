package TG

// tensor_test.go contains tests for functions in tensor.go

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

func Test_Index(t *testing.T) {
	// Test Index()
	A := RangeTensor([]int{10, 12, 14}, false)

	// Check that element at {2, 4, 3} is 395 in contiguous memory
	if A.Index([]int{2, 4, 3}) != 395 {
		t.Errorf("Index() failed. Expected Output: 395 --- Actual Output: %v", A.Index([]int{2, 4, 3}))
	}

}

func Test_Retrieve(t *testing.T) {

	// Test Retrieve()
	A := RangeTensor([]int{10, 12, 14}, false)

	// Check that element at {2, 4, 3} is 395 in contiguous memory
	if A.Get([]int{2, 4, 3}) != A.Data[395] {
		t.Errorf("Retrieve() failed. Expected Output: %v --- Actual Output: %v", A.Data[395], A.Get([]int{2, 4, 3}))
	}
}

func Test_UnravelIndex(t *testing.T) {

	// Test UnravelIndex()
	A := RangeTensor([]int{10, 12, 14}, false)

	// Check that element at {2, 4, 3} is 395 in contiguous memory
	Unraveled := A.UnravelIndex(395)

	// Check that the unraveled index is {2, 4, 3}
	if Unraveled[0] != 2 || Unraveled[1] != 4 || Unraveled[2] != 3 {
		t.Errorf("UnravelIndex() failed. Expected Output: [2, 4, 3] --- Actual Output: %v", Unraveled)
	}
}

func Test_Extract(t *testing.T) {

	// Init 2 range tensors, then 1 ones Tensor, then 2 range ones tensor
	ranges_1 := RangeTensor([]int{2, 10, 10}, true)
	randoms := OnesTensor([]int{1, 10, 10}, true)
	ranges_2 := RangeTensor([]int{2, 10, 10}, true)

	// Concatenate the tensors
	batched_tensors := ranges_1.Concat(randoms, 0)
	batched_tensors = batched_tensors.Concat(ranges_2, 0)

	// Extract the random tensor
	accessed_element := batched_tensors.GetBatchElement(2)

	if accessed_element.Sum_All() != 100 {
		t.Errorf("Extract() failed. Expected a 10x10 ones Tensor --- Actual Output:")
		Display_Matrix(accessed_element, false)
	}
}
