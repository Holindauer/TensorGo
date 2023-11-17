package GLA

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func Test_Tensor(t *testing.T) {

	fmt.Println()
	fmt.Println()
	fmt.Println("Now Testing Functions from tensor.go\n-------------------------------------")

	// Test Index()

	A := Range_Tensor([]int{10, 12, 14}, false)

	fmt.Println("A.Index(2, 4, 3) Expected Output 2*12*14 + 4*14 + 3 = 395 --- Actual Output: ", Index([]int{2, 4, 3}, A.Shape))
	fmt.Println()
	fmt.Println()

	// Test Retrieve()
	val := A.Retrieve([]int{2, 4, 3})

	fmt.Println("A.Retrieve(2, 4, 3) Expected Output: ", A.Data[395], " --- Actual Output: ", val)
	fmt.Println()
	fmt.Println()

	// Test UnravelIndex()
	fmt.Println("UnravelIndex(395, A.Shape) Expected Output: [2, 4, 3] --- Actual Output: ", UnravelIndex(395, A.Shape))

	// Test Acces()

	// Concatenate 4 range tensors, then 1 random Tensor, then 1 ones tensor
	ranges := Range_Tensor([]int{4, 10, 10}, true)
	randoms := RandFloat_Tensor([]int{1, 10, 10}, 0, 1, true)
	ones := Ones_Tensor([]int{1, 10, 10}, true)

	batched_tensors := ranges.Concat(randoms, 0)
	batched_tensors = batched_tensors.Concat(ones, 0)

	accessed_element := batched_tensors.Access(4)

	fmt.Println("Expected Shape of accessed_randoms: [10, 10] --- Actual Shape: ", accessed_element.Shape)

	fmt.Println("The following tensor should be a random tensor: ")
	Display_Matrix(accessed_element, false)

}
