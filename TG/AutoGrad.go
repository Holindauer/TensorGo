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
* @notice Value is a struct representing a node in the computational graph.
* @dev It holds the data for automatic differentiation.
* @param Data: The scalar value of this node.
* @param Grad: The gradient of this value with respect to the final output.
* @param BackwardFunc: A function that implements the backward pass for gradient computation.
* @param Children: References to the children in the computational graph (inputs to this node).
* @param Op: Descriptive string of the operation that created this node.
 */
type Value struct {
	Data         float64
	Grad         float64
	BackwardFunc func()
	Children     []*Value
	Op           string
}

/*
* @notice NewValue() is used to initialize a new Value node.
* @param data: The scalar value for this node.
* @param children: The child nodes of this node in the computational graph.
* @param op: The operation that this node represents.
* @return *Value: A pointer to the newly created Value.
 */
func NewValue(data float64, children []*Value, op string) *Value {
	return &Value{
		Data:         data,
		Children:     children,
		Op:           op,
		BackwardFunc: func() {}, // Default no-operation for backward pass.
	}
}

/*
* @notice Add() performs element-wise addition (on the level of individual scalars) and constructs the
* computational graph.
* @param other: The Value to be added.
* @return *Value: A new Value representing the result of the addition.
 */
func (v *Value) Add(other *Value) *Value {
	out := NewValue(v.Data+other.Data, []*Value{v, other}, "add")

	// Define the backward function for gradient computation using the chain rule.
	out.BackwardFunc = func() {
		v.Grad += out.Grad
		other.Grad += out.Grad
	}

	return out
}

/*
* @notice Mul() performs element-wise multiplication (on the level of individual scalars) and constructs the
* computational graph.
* @param other: The Value to be multiplied.
* @return *Value: A new Value representing the result of the multiplication.
 */
func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.Data*other.Data, []*Value{v, other}, "mul")

	// Define the backward function for gradient computation using the chain rule.
	out.BackwardFunc = func() {
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}

	return out
}

/*
* @notice ReLU() applies the Rectified Linear Unit activation function to the Value.
* @return *Value: A new Value representing the result of the ReLU operation.
 */
func (v *Value) ReLU() *Value {
	out := NewValue(math.Max(0, v.Data), []*Value{v}, "ReLU")

	// Define the backward function for the ReLU operation.
	out.BackwardFunc = func() {
		if v.Data > 0 {
			v.Grad += out.Grad
		}
	}

	return out
}

/*
* @notice BackwardPass() is used to compute the gradients for this node and all its ancestors in the computational graph.
* @dev This is a recursive function that calls the BackwardPass() function for all children of this node. The base case
* is when a node has no children.
 */
func (v *Value) BackwardPass() {

	// Initialize the gradient of the output node.
	if v.Grad == 0 {
		v.Grad = 1
	}

	// Run the backward function for this Value (defined in the operation that created this node).
	v.BackwardFunc()

	// Recursively apply the backward pass to all children.
	for _, child := range v.Children {
		child.BackwardPass()
	}
}
