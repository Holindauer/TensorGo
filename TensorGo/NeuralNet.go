package TG

import (
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
* @dev
* @dev The activation will be applied to each layer except the last.
* @param the layerNodeslice is used to specify the number of neurons in each layer.
 */
func MLP(inputFeatures int, layerNodes []int, activation string) *Layer {

	// create the first layer
	var layer *Layer = Linear(inputFeatures, layerNodes[0], activation, nil)

	// save the first layer pointer to return
	var firstLayer *Layer = layer

	// create the rest of the layers
	for i := 1; i < len(layerNodes); i++ {
		layer = Linear(layerNodes[i-1], layerNodes[i], activation, layer)
	}

	// return the first layer
	return firstLayer
}

/*
* @notice Linear() is a constructor function that is used to link together a series of
* Layers into a neural network in the style of pytorch.
*
 */
func Linear(inputFeatures int, layerNodes int, activation string, prev *Layer) *Layer {

	// @dev matrix eq of Perceptron; y = activation(x * W + b)
	var Weights *Tensor = new(Tensor) //(layers neurons x inputs) matrix of weights
	var Biases *Tensor = new(Tensor)  // vector of length layer neurons,

	Weights.Shape = []int{layerNodes, inputFeatures}
	Biases.Shape = []int{layerNodes}

	// slice of value pointers for the weights and biases
	Weights.DataReqGrad = make([]*Value, layerNodes*inputFeatures)
	Biases.DataReqGrad = make([]*Value, layerNodes)

	// Initialize the Value structs in weights and biases to random values
	for i := 0; i < layerNodes*inputFeatures; i++ {
		Weights.DataReqGrad[i] = NewValue(rand.Float64(), nil, "")
	}
	for i := 0; i < layerNodes; i++ {
		Biases.DataReqGrad[i] = NewValue(rand.Float64(), nil, "")
	}

	// create a new layer
	layer := &Layer{
		Neurons:    layerNodes,
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
	// return the new layer
	return layer
}
