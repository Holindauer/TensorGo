package TG

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func Test_SaveLoad(t *testing.T) {

	// Init a Tensor
	A := RandFloat64_Tensor([]int{2, 3}, 0, 1, false)

	// Save the Tensor to a JSON file
	A.Save_JSON("A.json")

	// Load the Tensor from the JSON file
	B := Load_JSON("A.json")

	// asert that A sums to the same value as B.
	if A.Sum_All() != B.Sum_All() {
		panic("A != B")
	}
}
