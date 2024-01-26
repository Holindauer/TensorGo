package main

import (
	"fmt"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

// Example usage and testing of the Value struct and its methods.
func main() {

	// Load Iris dataset
	var Iris *Tensor = LoadCSV("iris_dataset.csv", true)

	// Split targets from the features and turn on gradient tracking
	features, targets := Gradify(Iris.Slice("1:, :4")), Gradify(Iris.Slice("1:, 4:"))

	fmt.Println("Features shape: ", features.Shape, " Targets shape: ", targets.Shape)

	// Create a new multi layer perceptron
	var mlp *Layer = MLP(
		4,                                // input features
		[]int{16, 16, 3},                 // neurons per layer
		[]string{"relu", "relu", "relu"}, // activations
	)

	// Print a model summary
	Summary(mlp)

	epochs := 10
	lr := 0.001

	for i := 0; i < epochs; i++ {

		fmt.Println("\nEpoch: ", i+1)

		// forward pass
		var output *Tensor = mlp.Forward(features)
		fmt.Println("Output shape: ", output.Shape)

		mlp.ZeroGrad()

		fmt.Println("Output shape: ", output)

		//make prediction
		var pred *Tensor = ArgmaxVector(output, true)

		fmt.Println("Pred shape: ", pred.Shape)

		var loss *Value = CrossEntropy(pred, targets)

		loss.Backward()

		mlp.Step(lr)

	}

}
