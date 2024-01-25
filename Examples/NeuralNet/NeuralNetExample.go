package main

import (
	"fmt"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

// Example usage and testing of the Value struct and its methods.
func main() {

	// Load Iris dataset
	var Iris *Tensor = LoadCSV("iris_dataset.csv", true)
	fmt.Println("Iris Dataset shape: ", Iris.Shape)

	//Seperate the targets from the features
	features, targets := Iris.Slice(":, :4"), Iris.Slice(":, 4:")

	fmt.Println("Features shape: ", features.Shape, " Targets shape: ", targets.Shape)

	// Split data into train and test sets
	trainFeatures, trainTargets := Gradify(features.Slice(":120, :")), Gradify(targets.Slice(":120, :"))
	testFeatures, testTargets := features.Slice("120:, :"), targets.Slice("120:, :")

	fmt.Println("\nTrain Features shape: ", trainFeatures.Shape)
	fmt.Println("Train Targets shape: ", trainTargets.Shape)
	fmt.Println("\nTest Features shape: ", testFeatures.Shape)
	fmt.Println("Test Targets shape: ", testTargets.Shape)

	// Create a new MLP
	var mlp *Layer = MLP(
		4,                                // input features
		[]int{16, 16, 3},                 // neurons per layer
		[]string{"relu", "relu", "relu"}, // activations
	)

	// Print a summary of the MLP
	Summary(mlp)

	// var output *Tensor = mlp.Forward(trainFeatures)
	// fmt.Println("Output shape: ", output.Shape)

	epochs := 10
	lr := 0.001

	// THE ISSUE IS IN THE FORWARD PASS OF THE MLP, NOT THE ARGMAX

	for i := 0; i < epochs; i++ {

		fmt.Println("\nEpoch: ", i+1)

		// forward pass
		var output *Tensor = mlp.Forward(trainFeatures)
		fmt.Println("Output shape: ", output.Shape)

		mlp.ZeroGrad()

		fmt.Println("Output shape: ", output)

		//make prediction
		var pred *Tensor = ArgmaxVector(output, true)

		fmt.Println("Pred shape: ", pred.Shape)

		var loss *Value = CrossEntropy(pred, trainTargets)

		loss.Backward()

		mlp.Step(lr)

	}

}
