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
func MLP(inputFeatures int, layerNodes []int, activations []string) *Layer {

	// create the first layer
	var layer *Layer = Linear(inputFeatures, layerNodes[0], activations[0], nil)

	// save the first layer pointer to return
	var firstLayer *Layer = layer

	// create the rest of the layers
	for i := 1; i < len(layerNodes); i++ {
		layer = Linear(layerNodes[i-1], layerNodes[i], activations[i], layer)
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

	// contiguous slice of value pointers for the weights and biases
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

/*
* @notice Forward() is a function that is used to perform a forward pass through a multilayer perceptron.
* @dev The forward pass by default works on batched Tensors. The batch size is the first dimension of the Tensor.
* @param Batch: A batch of Tensors to be passed through the MLP.
* @param Net: A pointer to the first layer in the MLP linked list of Layer structs.
 */
func (Net *Layer) Forward(Batch *Tensor) *Tensor {

	if !Batch.Batched {
		panic("Within Forward(): Dataset must be batched")
	}

	var x *Tensor = Batch

	// Iterate layers in the network
	for Net != nil {

		// Multiply weights, add biases
		x = MatMulGrad(x, Net.Weights, true) // <-- MatrixOps.go
		x = AddGrad(x, Net.Biases, true)     // <-- ClosedBinaryOps.go

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
			// Apply Softmax to all elements in the hidden state
			x = x.Softmax() // Softmax() requires the entire vector
		}

		// next layer
		Net = Net.Next
	}

	return x
}

/*
* @notice ZeroGrad() is a function that is used to zero out the gradients of all the Value structs in all layers
* in an MLP linked list of Layer structs.
* @param layer: The first layer in the MLP linked list of Layer structs.
 */
func (layer *Layer) ZeroGrad() {

	var numWeights int
	var numBiases int

	// print the number of neurons in each layer
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
