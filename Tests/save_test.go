package TG

import (
	"fmt"

	"testing"

	. "github.com/Holindauer/Tensor-Go.git/TG"
)

func Test_Save(t *testing.T) {
	fmt.Println("Testing save.go")

	fmt.Println("Testing MarshalTensor function")

	tensor := Tensor{
		Shape:    []int{1, 2, 3},
		Data:     []float64{1.001, 2.13, 3.0},
		BoolData: []bool{true, false, true},
		Batched:  true,
	}

	json_tensor := MarshalTensor(&tensor)
	fmt.Println(json_tensor)

	fmt.Println()
	fmt.Println()

	fmt.Println("Testing Save function")
	// Save("test.json", &json_tensor)
}
