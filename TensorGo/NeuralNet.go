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

/*
* @notice Forward() is a function that is used to perform a forward pass through a multilayer perceptron.
* @dev The forward pass by default works on batched Tensors. This
 */
func (Net *Layer) Forward(Batch *Tensor) *Tensor {

	if !Batch.Batched {
		panic("Within Forward(): Dataset must be batched")
	}

	var x *Tensor = Batch

	// Iterate layers in the network
	for Net != nil {

		// Matrix multiplication and bias addition
		// Assuming a function MatMul for matrix multiplication and Add for addition
		x = MatMulGrad(x, Net.Weights, true)
		x = AddGrad(x, Net.Biases, true)

		// Apply activation function for the layer
		switch Net.Activation {
		case "relu":
			// Apply ReLU to all elements in the hidden state
			for i := 0; i < len(x.DataReqGrad); i++ {
				x.DataReqGrad[i] = x.DataReqGrad[i].ReLU()
			}
		}

		// next layer
		Net = Net.Next
	}

	return x
}

//===================================================================================================================== Gradient Tracked Elementwise Tensor Addition

type EWAdditionGradTracked struct{}

// @notice this method implements gradient tracked addition of 2 Value structs
func (ea EWAdditionGradTracked) ExecuteElementwiseOp(a, b *Value) *Value {
	return a.Add(b) // <--- Value Method from AutoGrad.go
}

func (ba EWAdditionGradTracked) Execute(tensors ...*Tensor) *Tensor {
	A, B := tensors[0], tensors[1]
	return ElementwiseOp(A, B, EWAddition{})
}

/*
* @notice AddGrad() implements elementwise Tensor addition with gradient tracking.
* @dev Gradient tracked data is stored in the DataReqGrad slice of the Tensor struct, different than the Data slice of just sclalars.
* @dev The DataReqGrad slice stores pointers to Value structs, which store the computational graph and gradient for
* reverse mode autodiff.
 */
func AddGrad(A *Tensor, B *Tensor, batching bool) *Tensor {

	addGrad := EWAdditionGradTracked{} // Create an instance of EWAddition

	if batching {
		return BatchedOperation(addGrad, A, B) // batched op
	}
	return addGrad.Execute(A, B) // single op
}

//===================================================================================================================== Gradient Tracked Matrix Mulitplication

type MatMulGradOp struct{}

// Implementing the Execute method of IBatching interface
func (op MatMulGradOp) Execute(tensors ...*Tensor) *Tensor {
	// Assumes tensors length will be 2 for matrix multiplication
	A, B := tensors[0], tensors[1]

	// In case of Matrix Vector Multiplication
	if len(B.Shape) == 1 {
		B = B.Add_Singleton(0)
	}

	// Check that the two Tensors are compatible for matrix multiplication
	Check_MatMul_Compatibility(A, B)

	C := Zero_Tensor([]int{A.Shape[0], B.Shape[1]}, false)
	var sum *Value

	for row := 0; row < C.Shape[0]; row++ {
		for col := 0; col < C.Shape[1]; col++ {

			sum = NewValue(0.0, nil, "")

			for k := 0; k < A.Shape[1]; k++ {

				// Gradient Tracked Dot Product
				elementA := A.DataReqGrad[A.Index([]int{row, k})]
				elementB := B.DataReqGrad[B.Index([]int{k, col})]

				// grad tracked multiplication
				var mul *Value = elementA.Mul(elementB)

				sum = sum.Add(mul)
			}

			C.DataReqGrad[C.Index([]int{row, col})] = sum
		}
	}

	return C
}

func MatMulGrad(A *Tensor, B *Tensor, batching bool) *Tensor {

	matmul := MatMulOp{} // Create an instance of Batched_Matmul

	if batching {
		// If batching is true, call BatchedOperation directly
		return BatchedOperation(matmul, A, B)
	} else {
		// If batching is false, call the Execute method directly
		return matmul.Execute(A, B)
	}
}

//=====================================================================================================================

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
