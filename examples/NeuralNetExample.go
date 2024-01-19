package main

import (
	"fmt"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

// Example usage and testing of the Value struct and its methods.
func main() {

	var mlp *Layer = MLP(
		3,                 // input features
		[]int{8, 8, 8, 1}, // neurons per layer
		"relu",            // activation function
	)

	i := 1

	// print the number of neurons in each layer
	for mlp != nil {
		fmt.Println("Layer: ", i, " Neurons: ", mlp.Neurons, " Activation: ", mlp.Activation, " Weights shape: ", mlp.Weights.Shape, " Biases shape: ", mlp.Biases.Shape)
		mlp = mlp.Next
		i++
	}

	// zero out the gradients
	mlp.ZeroGrad()

	// make a dummy input
	var dummyInput *Tensor = new(Tensor)
	dummyInput.Shape = []int{1, 3}
	dummyInput.DataReqGrad = make([]*Value, 3)
	dummyInput.Batched = true
	for i := 0; i < 3; i++ {
		dummyInput.DataReqGrad[i] = NewValue(1, nil, "")
	}

	// forward pass
	var output *Tensor = mlp.Forward(dummyInput)

	// print the output
	fmt.Println("Output: ", output.DataReqGrad[0].Scalar)

	// backward pass
	output.DataReqGrad[0].Backward()
}
