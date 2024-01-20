package TG

/*
* @notice This is a re-implementation of Andrej Karpathy's MircroGrad autograd engine, adapted for Go / Tensor-Go.
* @notice AutoGrad.go implements a directed acyclic graph for reverse mode autodiff on individual scalars. This graph is
* used to implement backpropagation.
 */

import (
	"math"
)

/*
* @notice The Value struct represents a node in the computational graph and is used to construct a
* directed acyclic graph for reverse mode automatic differentiation.
* @param Scalar: The scalar value of this node.
* @param Grad: The gradient of this value with respect to the final output.
* @param _backward: A function that implements the backward pass for gradient computation.
* @param _prev: References to the previous nodes in the computational graph.
* @param Op: Descriptive string of the operation that created this node.
 */
type Value struct {
	Scalar    float64
	Grad      float64
	_backward func()
	_prev     []*Value
	Op        string
}

/*
* @notice NewValue initializes and returns a new Value node.
* @param item: The scalar value for this node.
* @param _prev: The previous nodes of this node in the computational graph.
* @param _op: The operation that this node represents.
* @return A pointer to the newly created Value.
 */
func NewValue(scalar float64, _prev []*Value, _op string) *Value {
	return &Value{
		Scalar:    scalar,
		_prev:     _prev,
		Op:        _op,
		_backward: func() {},
	}
}

/*
* @notice Backward uses the _backward() method of each node to compute the gradients for this node and all its
* ancestors in the computational graph. This is done using a topological sort to ensure that no ancestor node
* is processed before its children. This is necessary to handle when values are used multiple times in the
* computational graph.
 */
func (v *Value) Backward() {

	//
	topo := TopologicalSort(v)
	v.Grad = 1

	for _, node := range topo {
		node._backward()
	}
}

/*
* @notice TopologicalSort performs a topological sort on the nodes of the computational graph.
* @dev It ensures that the nodes are processed in the correct order during the backward pass.
* @param v: The starting Value node for the topological sort.
* @return A slice of *Value in the order they should be processed.
 */
func TopologicalSort(v *Value) []*Value {

	// Make a slice of *Value to store the topologically sorted nodes.
	var topo []*Value

	// Make a map from *Value to bool to track which nodes have been visited.
	visited := make(map[*Value]bool)

	// Define a recursive function to build the topologically sorted slice.
	var buildTopo func(*Value)
	buildTopo = func(node *Value) {

		if !visited[node] { // <--- Mark unvisited nodes as visited
			visited[node] = true

			// recursive call on children
			for _, child := range node._prev {
				buildTopo(child)
			}

			// append the node to the topologically sorted slice
			topo = append(topo, node)
		}
	}

	// Kickstart the recursive topological sort
	buildTopo(v)

	// return the reversed slice
	return reverse(topo)
}

/*
* @notice reverse reverses the slice of *Value. It's used in the topological sort to reverse the order of the
* sorted nodes. This is necessary because the topological sort is implemented recursively, and the nodes are
* appended to the slice in the reverse order of the topological sort.
 */
func reverse(values []*Value) []*Value {
	for i, j := 0, len(values)-1; i < j; i, j = i+1, j-1 {
		values[i], values[j] = values[j], values[i]
	}
	return values
}

// ===================================================================================================================== Value Methods Below

/*
* @notice Add performs addition with another Value, constructing the computational graph between the two.
* @dev The out._backward function is defined here, which implements the chain rule for addition at this
* node in the computational graph. Note that because this is reverse mode autodiff, the chain rule is
* implemented in reverse order and relies on the previously computed gradients.
* @param v: The Value to be added.
* @param other: The other Value to be added.
* @return A pointer to a new Value representing the result of the addition.
 */
func (v *Value) Add(other *Value) *Value {
	out := NewValue(v.Scalar+other.Scalar, []*Value{v, other}, "add")

	/// @dev the chain rule for z = x + y is dz/dx = 1 and dz/dy = 1
	out._backward = func() {
		v.Grad += out.Grad
		other.Grad += out.Grad
	}

	return out
}

/*
* @notice Mul performs element-wise multiplication with another Value and constructs the computational graph.
* @dev The out._backward function is defined here, which implements the chain rule for multiplication at this
* node in the computational graph. Note that because this is reverse mode autodiff, the chain rule is
* implemented in reverse order and relies on the previously computed gradients.
* @param v: The Value to be multiplied.
* @param other: The other Value to be multiplied.
* @return A pointer to a new Value representing the result of the multiplication.
 */
func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.Scalar*other.Scalar, []*Value{v, other}, "mul")

	/// @dev the chain rule for z = x * y is dz/dx = y and dz/dy = x
	out._backward = func() {
		v.Grad += other.Scalar * out.Grad
		other.Grad += v.Scalar * out.Grad
	}

	return out
}

/*
* @notice Div performs element-wise division with another Value and constructs the computational graph.
* @dev The out._backward function is defined here, which implements the chain rule for division at this
* node in the computational graph. Note that because this is reverse mode autodiff, the chain rule is
* implemented in reverse order and relies on the previously computed gradients.
* @param v: The Value to be divided.
* @param other: The other Value to be divided.
* @return A pointer to a new Value representing the result of the division.
 */
func (v *Value) Div(other *Value) *Value {

	// compute x / y
	divVal := v.Scalar / other.Scalar

	// create a new Value struct for the division, feeding in the ancestor node
	out := NewValue(divVal, []*Value{v, other}, "div")

	out._backward = func() { // z = x / y --> dz/dx = 1/y and dz/dy = -x/y^2
		v.Grad += 1 / other.Scalar * out.Grad
		other.Grad += -v.Scalar / (other.Scalar * other.Scalar) * out.Grad
	}

	return out
}

/*
* @notive Exp() applies the exponential function to the Value.
* @dev The out._backward function is defined here, which implements the chain rule for exp at this
* node in the computational graph. Note that because this is reverse mode autodiff, the chain rule is
* implemented in reverse order and relies on the previously computed gradients.
* @param v: The Value to be exponentiated.
* @return A pointer to a new Value representing the result of the exp operation.
 */
func (v *Value) Exp() *Value {
	// compute e^x
	expVal := math.Exp(v.Scalar)

	// create a new Value struct for the exp, feeding in the ancestor node
	out := NewValue(expVal, []*Value{v}, "exp")

	out._backward = func() {
		v.Grad += expVal * out.Grad // d(e^x)/dx = e^x
	}

	return out
}

/*
* @notice ReLU applies the Rectified Linear Unit activation function to the Value.
* @dev The out._backward function is defined here, which implements the chain rule for ReLU at this
* node in the computational graph. Note that because this is reverse mode autodiff, the chain rule is
* implemented in reverse order and relies on the previously computed gradients.
* @param v: The Value to be ReLU'd.
* @return A pointer to a new Value representing the result of the ReLU operation.
 */
func (v *Value) ReLU() *Value {
	out := NewValue(math.Max(0, v.Scalar), []*Value{v}, "ReLU")

	/// @dev the chain rule for z = ReLU(x) is dz/dx = 1 if x > 0 and dz/dx = 0 if x <= 0
	out._backward = func() {
		if v.Scalar > 0 {
			v.Grad += out.Grad
		}
	}

	return out
}

/*
* @notice Sigmoid applies the Sigmoid activation function to the Value.
* @dev Sigmoid is defined as 1 / (1 + e^-x)
* @dev The derivative of sigmoid is sigmoid * (1 - sigmoid)
* @param v: The Value struct to be Sigmoid'd.
* @return A pointer to a new Value struct representing the result of the Sigmoid operation.
 */
func (v *Value) Sigmoid() *Value {
	// compute sigmoid
	sigmoid := 1 / (1 + math.Exp(-v.Scalar))

	// create a new Value struct for the sigmoid
	out := NewValue(sigmoid, []*Value{v}, "sigmoid")

	out._backward = func() {
		// The derivative of sigmoid is sigmoid * (1 - sigmoid)
		v.Grad += out.Scalar * (1 - out.Scalar) * out.Grad
	}

	return out
}

/*
* @notice Softmax applies the Softmax activation function to the Value.
* @dev unlike .ReLU() and .Sigmoid(), which are applied to individual values, .Softmax() requires all values in a Vector to compute.
* As such, .Softmax() accepts a Tensor as input and returns a Tensor as output.
* @dev Backward() func is absent because Softmax is broken into multiple steps to which the gradients are computed at each steps.
 */
func (x *Tensor) Softmax() *Tensor {
	// Check if the tensor is a vector
	if len(x.Shape) != 1 {
		panic("Softmax can only be applied to vectors.")
	}

	// Create a slice of *Value to store the e^value for each value in the vector
	var exps []*Value = make([]*Value, len(x.DataReqGrad))

	// Create a new Value to store the sum of all e^values
	var sumExp *Value = NewValue(0.0, nil, "")

	// Calculate e^value for each value and sum them
	for i, value := range x.DataReqGrad {
		expVal := value.Exp()
		exps[i] = expVal
		sumExp = sumExp.Add(expVal)
	}

	// Normalize each e^value by the sum of all e^values
	softmaxValues := make([]*Value, len(x.DataReqGrad))
	for i, expVal := range exps {
		softmaxValues[i] = expVal.Div(sumExp)
	}

	// Create a new Tensor for the softmax output
	softmaxTensor := &Tensor{
		Shape:       []int{len(softmaxValues)},
		DataReqGrad: softmaxValues,
		Batched:     x.Batched,
	}

	return softmaxTensor
}
