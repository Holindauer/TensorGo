package TG

import (
	"fmt"
	"math/rand"
)

/*
* @notice NeuralNet.go implements neural networks (only multilayer perceptrons at the moment) in Tensor-Go.
* @dev Neural Net Layers are constructed as a linked list of Layer structs.
* @dev Each Layer Struct is composed of Tensors that store individual weights and biases in Value structs. Within
* the Tensor's DataReqGrad slice, which stores Value pointers. Each Value struct stores the computational graph
* and gradient for reverse mode automatic differentiation.
 */

/*
* @notice The Layer Struct represents a single layer in a neural network.
* @dev The weight matrix and bias vector are stored as Tensors
* @dev Each Layer is a node in a bidirectional linked list of Layers.
* @dev The Activation function is a string that represents the activation function to be used
 */
type Layer struct {
	Neurons    int
	Weights    *Tensor
	Biases     *Tensor
	Activation string
	Next       *Layer // <-- Next and Prev Nodes
	Prev       *Layer
}

/*
* @notice MLP() is a constructor function that is used to create a multilayer perceptron.
* @dev The MLP is constructed as a linked list of Layer structs.
* @dev The activation will be applied to each layer in the list.
* @param layerNodeslice is used to specify the number of neurons in each layer.
 */
func MLP(inputFeatures int, layerNodes []int, activations []string) *Layer {

	// create the first layer
	var layer *Layer = Linear(inputFeatures, layerNodes[0], activations[0], nil)

	// save the first layer pointer to return
	var firstLayer *Layer = layer

	// create and link together the rest of the layers
	for i := 1; i < len(layerNodes); i++ {
		layer = Linear(layerNodes[i-1], layerNodes[i], activations[i], layer)
	}

	return firstLayer
}

/*
* @notice Linear() is a constructor function that creates and connects a new linked list node, the Layer struct
* @dev if there is a previous node input as argument, the previous node is connected behind it.
* @param inputFeatures: The number of input features moving into the Layer
* @param layerNodes: The number of neurons in the Layer
 */
func Linear(inputFeatures int, layerNeurons int, activation string, prev *Layer) *Layer {

	// @dev matrix eq of Perceptron; y = activation(Wx + b)
	var Weights *Tensor = new(Tensor)
	var Biases *Tensor = new(Tensor)

	// set shapes
	Weights.Shape = []int{layerNeurons, inputFeatures}
	Biases.Shape = []int{layerNeurons}

	// contiguous slice of value pointers for the weights and biases
	Weights.DataReqGrad = make([]*Value, layerNeurons*inputFeatures)
	Biases.DataReqGrad = make([]*Value, layerNeurons)

	// Initialize the Value structs in weights and biases to random values
	for i := 0; i < layerNeurons*inputFeatures; i++ {
		Weights.DataReqGrad[i] = NewValue(rand.Float64(), nil, "") // TODO: allow custom initialization
	}
	for i := 0; i < layerNeurons; i++ {
		Biases.DataReqGrad[i] = NewValue(rand.Float64(), nil, "")
	}

	// Set RequireGrad to true for the weights and biases
	Weights.RequireGrad, Biases.RequireGrad = true, true

	// create a new layer
	layer := &Layer{
		Neurons:    layerNeurons,
		Weights:    Weights,
		Biases:     Biases,
		Activation: activation,
		Next:       nil,
		Prev:       prev,
	}

	// link the previous layer to this layer
	if prev != nil {
		prev.Next = layer
	}

	return layer
}

/*
* @notice Forward() is a function that is used to perform a forward pass through a multilayer perceptron.
* @dev The forward pass by default works on batched Tensors. The batch size is the first dimension of the Tensor.
* @dev we are manually splitting up the batch here instead of calling MatMulGrad w/ batching set to true because
* the batch size of the input will not match to the batch size of the weights (which there is none)
* @param Batch: A batch of Tensors to be passed through the MLP.
* @param Net: A pointer to the first layer in the MLP linked list of Layer structs.
 */
func (Net *Layer) Forward(Batch *Tensor) *Tensor {

	if !Batch.Batched {
		panic("Within Forward(): Dataset must be batched")
	}
	if !Batch.RequireGrad {
		panic("Within Forward(): Dataset must require gradient")
	}
	if len(Batch.Shape) != 2 {
		panic("Within Forward(): MultiLayer Perceptrons Require input Tensor to be shaped: [BatchSize, InputFeatures]")
	}

	// Add singleton dimension to the front of the batch as concat dim
	var x *Tensor = Batch.Add_Singleton(2)

	// Iterate layers in the network
	for Net != nil {

		// multiply weights and add biases
		x = weightsBiases(x, Net.Weights, Net.Biases)

		// Apply activation function for the layer
		switch Net.Activation {
		case "relu":
			// Apply ReLU to all elements in the hidden state
			for i := 0; i < len(x.DataReqGrad); i++ {
				x.DataReqGrad[i] = x.DataReqGrad[i].ReLU()
			}
		case "sigmoid":
			// Apply Sigmoid to all elements in the hidden state
			for i := 0; i < len(x.DataReqGrad); i++ {
				x.DataReqGrad[i] = x.DataReqGrad[i].Sigmoid()
			}
		case "softmax":

			// Apply Softmax to each batch element individually
			x = x.Remove_Dim(0, 0).Softmax()
			x.Shape = []int{1, x.Shape[0]}

			// Iterate through the batch of inputs, applying Softmax to each element, concatenating the results
			for i := 1; i < x.Shape[0]; i++ {
				tempOutput := x.Remove_Dim(0, i).Softmax()
				tempOutput.Shape = []int{1, tempOutput.Shape[0]}
				x = x.Concat(tempOutput, 0)
			}
		}

		Net = Net.Next
	}

	return x
}

/*
* @notice this helper function applies gradient tracked matmul adn addition for weights/biases to each element in a batch of inputs for an mlp.
* @param batch: a batch of inputs
* @param weights: the weights of the layer
 */
func weightsBiases(batch *Tensor, weights, bias *Tensor) *Tensor {

	// Anon func used to extract a single element from a 2D batch of inputs
	getElement := func(batch *Tensor, index int) *Tensor {
		batchElement := ZeroTensor([]int{batch.Shape[1], 1}, false)
		for i := 0; i < batch.Shape[1]; i++ {
			batchElement.DataReqGrad[i] = batch.DataReqGrad[i+index]
		}
		return batchElement
	}

	// Apply matrix multiplication to each element in the batch of inputs, concatenating the results
	outputAccumulator := MatMulGrad(weights, getElement(batch, 0), false) // <-- MatrixOps.go

	// Iterate through the batch of inputs, applying matrix multiplication to each element
	for i := 1; i < batch.Shape[0]; i++ {

		// multiply weights
		tempOutput := MatMulGrad(weights, getElement(batch, i), false) // <-- MatrixOps.go

		// add biases
		for j := 0; j < len(tempOutput.DataReqGrad); j++ {
			tempOutput.DataReqGrad[j] = tempOutput.DataReqGrad[j].Add(bias.DataReqGrad[j]) // <-- AutoGrad.go
		}

		// append the results to the output accumulator
		outputAccumulator.DataReqGrad = append(outputAccumulator.DataReqGrad, tempOutput.DataReqGrad...)
	}
	// set the shape of the output accumulator to the correct shape post matmul
	outputAccumulator.Shape = []int{batch.Shape[0], outputAccumulator.Shape[0]}

	return outputAccumulator
}

/*
* @notice ZeroGrad() is a function that is used to zero out the gradients of all the Value structs in all layers
* in an MLP linked list of Layer structs.
* @param layer: The first layer in the MLP linked list of Layer structs.
 */
func (layer *Layer) ZeroGrad() {

	var numWeights int
	var numBiases int

	for layer != nil {

		// get the number of weights and biases in this layer
		numWeights = layer.Weights.Shape[0] * layer.Weights.Shape[1]
		numBiases = layer.Biases.Shape[0]

		// zero weight grads
		for i := 0; i < numWeights; i++ {
			layer.Weights.DataReqGrad[i].Grad = 0.0
		}

		// zero bias grads
		for i := 0; i < numBiases; i++ {
			layer.Biases.DataReqGrad[i].Grad = 0.0
		}

		// next node
		layer = layer.Next
	}
}

/*
* @notice Step() applies the gradient descent learning rule to the weights and biases of an MLP.
* @dev for Step() to be used, first the gradient must be computed for each Value struct in the MLP. This is done by
* calling .Backward() on the output of the MLP, which will call .Backward() on each Value struct in the MLP.
* @param learningRate: The learning rate to be used in the gradient descent learning rule.
* @param layer: The first layer in the MLP linked list of Layer structs.
 */
func (layer *Layer) Step(learningRate float64) {

	// Iterate through each layer of the MLP
	for layer != nil {

		// Update weights
		numWeights := len(layer.Weights.DataReqGrad)
		for i := 0; i < numWeights; i++ {
			// Gradient descent update rule: W = W - learningRate * dW
			layer.Weights.DataReqGrad[i].Scalar -= learningRate * layer.Weights.DataReqGrad[i].Grad
		}

		// Update biases
		numBiases := len(layer.Biases.DataReqGrad)
		for i := 0; i < numBiases; i++ {
			// Gradient descent update rule: b = b - learningRate * db
			layer.Biases.DataReqGrad[i].Scalar -= learningRate * layer.Biases.DataReqGrad[i].Grad
		}

		// Proceed to the next layer
		layer = layer.Next
	}
}

/*
* @notice Summary() is a function that prints a summary of the MLP.
* @param layer: The first layer in the MLP linked list of Layer structs.
 */
func Summary(mlp *Layer) {
	i := 0
	// print the number of neurons in each layer
	fmt.Println("\nMLP Summary: ")
	for mlp != nil {
		fmt.Println("Layer: ", i, " Neurons: ", mlp.Neurons, " Activation: ", mlp.Activation, " Weights shape: ", mlp.Weights.Shape, " Biases shape: ", mlp.Biases.Shape)
		mlp = mlp.Next
		i++
	}

	fmt.Println()
}
