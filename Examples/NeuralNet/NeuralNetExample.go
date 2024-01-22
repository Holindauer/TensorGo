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

	//feature := Gradify(features.Remove_Dim(0, 2).Reshape([]int{4, 1}, false))

	// Split data into train and test sets
	trainFeatures, trainTargets := Gradify(features.Slice(":120, :")), Gradify(targets.Slice(":120, :"))
	testFeatures, testTargets := features.Slice("120:, :"), targets.Slice("120:, :")

	fmt.Println("\nTrain Features shape: ", trainFeatures.Shape)
	fmt.Println("Train Targets shape: ", trainTargets.Shape)
	fmt.Println("\nTest Features shape: ", testFeatures.Shape)
	fmt.Println("Test Targets shape: ", testTargets.Shape)

	// Create a new MLP
	var mlp *Layer = MLP(
		4,                                   // input features
		[]int{16, 16, 3},                    // neurons per layer
		[]string{"relu", "relu", "softmax"}, // activations
	)

	// Print a summary of the MLP
	Summary(mlp)

	var output *Tensor = mlp.Forward(trainFeatures)
	fmt.Println("\nOutput shape: ", output.Shape)

	// for i := 0; i < trainFeatures.Shape[0]; i++ {

	// 	for j := 0; j < 4; j++ {
	// 		fmt.Print("Feature: ", trainFeatures.DataReqGrad[i+j])
	// 	}

	// 	fmt.Println("\nTrain Targets: ", trainTargets.Data[i])
	// }

	// i := 1

	// var output *Tensor = mlp.Forward(trainFeatures)
	// fmt.Println("Output shape: ", output.Shape)

	//epochs := 10
	//lr := 0.001

	// for i := 0; i < epochs; i++ {

	// 	// forward pass
	// 	var output *Tensor = mlp.Forward(trainFeatures)
	// 	fmt.Println("Output shape: ", output.Shape)

	// 	mlp.ZeroGrad()

	// 	fmt.Println("Output shape: ", output.Shape, "lr", lr)

	// 	// make prediction
	// 	var pred *Tensor = output.ArgmaxVector(true)
	// 	fmt.Println("Pred shape: ", pred.Shape)

	// 	var loss *Value = CrossEntropy(pred, trainTargets)

	// 	loss.Backward()

	// 	mlp.Step(lr)

	// 	fmt.Println("Epoch: ", i)

	// }

}
