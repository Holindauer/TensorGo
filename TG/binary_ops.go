package TG

import "fmt"

/*
* @notice binary_ops.go contains functions that accept two tensor of the same shape and return a single tensor with that shapes
 */

// =========================================================================================================== Elementwise Tensor Operations gereralization

// This interace is used to generalize elementwise tensor operations on the level of individual elements
type ElementWiseOp interface {
	EWOp(a, b float64) float64
}

// This function is a generalization of elementwise tensor operations. It takes in two tensors and an Element_Operation.
func Elementwise_Operation(A *Tensor, B *Tensor, op ElementWiseOp) *Tensor {

	if !Same_Shape(A, B) {
		fmt.Println("Within Elementwise_Operation(): -- A.Shape:", A.Shape, "B.Shape:", B.Shape)
		panic("Within Elementwise_Operation(): Tensors must have the same shape")
	}

	C := Zero_Tensor(A.Shape, false)
	for i := 0; i < len(A.Data); i++ {
		C.Data[i] = op.EWOp(A.Data[i], B.Data[i]) // perform operation with elements
	}
	return C // <-- pointer
}

//=========================================================================================================== Elementwise Tensor Addition

// Define a struct that implements the Element_Operation interface
type EWAddition struct{}

func (ea EWAddition) EWOp(a, b float64) float64 {
	return a + b
}

func (ba EWAddition) Execute(tensors ...*Tensor) *Tensor {
	A, B := tensors[0], tensors[1]
	return Elementwise_Operation(A, B, EWAddition{})
}

// This function performs elementwise addition on two tensors with optional batching
func Add(A *Tensor, B *Tensor, batching bool) *Tensor {

	ewAddition := EWAddition{} // Create an instance of EWAddition

	if batching {
		return BatchedOperation(ewAddition, A, B) // batched op
	}
	return ewAddition.Execute(A, B) // single op
}

//=========================================================================================================== Elementwise Tensor Subtraction

// Define a struct that implements the Element_Operation interface
type EWSubtraction struct{}

func (es EWSubtraction) EWOp(a, b float64) float64 {
	return a - b
}

func (ba EWSubtraction) Execute(tensors ...*Tensor) *Tensor {
	A, B := tensors[0], tensors[1]
	return Elementwise_Operation(A, B, EWSubtraction{})
}

// This function performs elementwise addition on two tensors with optional batching
func Subtract(A *Tensor, B *Tensor, batching bool) *Tensor {

	ewSubtraction := EWSubtraction{} // Create an instance of EWSubtraction

	if batching {
		return BatchedOperation(ewSubtraction, A, B) // batched op
	}
	return ewSubtraction.Execute(A, B) // single op
}

//=========================================================================================================== Elementwise Tensor Multiplicatio

// Define a struct that implements the Element_Operation interface
type EWMultiplication struct{}

func (es EWMultiplication) EWOp(a, b float64) float64 {
	return a * b
}

func (ba EWMultiplication) Execute(tensors ...*Tensor) *Tensor {
	A, B := tensors[0], tensors[1]
	return Elementwise_Operation(A, B, EWMultiplication{})
}

// This function performs elementwise addition on two tensors with optional batching
func Multiply(A *Tensor, B *Tensor, batching bool) *Tensor {

	ewMultiplication := EWMultiplication{} // Create an instance of EWMultiplication

	if batching {
		return BatchedOperation(ewMultiplication, A, B) // batched op
	}
	return ewMultiplication.Execute(A, B) // single op
}
