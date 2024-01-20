package TG

/*
* @notice closed_binary_ops.go contains functions that accept two tensor of the same shape and return a single tensor with that shapes
 */

//===================================================================================================================== Elementwise Tensor Addition

// Define a struct that implements the Element_Operation interface
type EWAddition struct{}

func (ea EWAddition) ExecuteElementwiseOp(a, b float64) float64 {
	return a + b
}

func (ba EWAddition) Execute(tensors ...*Tensor) *Tensor {
	A, B := tensors[0], tensors[1]
	return ElementwiseOp(A, B, EWAddition{})
}

// This function performs elementwise addition on two tensors with optional batching
func Add(A *Tensor, B *Tensor, batching bool) *Tensor {

	ewAddition := EWAddition{} // Create an instance of EWAddition

	if batching {
		return BatchedOperation(ewAddition, A, B) // batched op
	}
	return ewAddition.Execute(A, B) // single op
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

//===================================================================================================================== Elementwise Tensor Subtraction

// Define a struct that implements the Element_Operation interface
type EWSubtraction struct{}

func (es EWSubtraction) ExecuteElementwiseOp(a, b float64) float64 {
	return a - b
}

func (ba EWSubtraction) Execute(tensors ...*Tensor) *Tensor {
	A, B := tensors[0], tensors[1]
	return ElementwiseOp(A, B, EWSubtraction{})
}

// This function performs elementwise addition on two tensors with optional batching
func Subtract(A *Tensor, B *Tensor, batching bool) *Tensor {

	ewSubtraction := EWSubtraction{} // Create an instance of EWSubtraction

	if batching {
		return BatchedOperation(ewSubtraction, A, B) // batched op
	}
	return ewSubtraction.Execute(A, B) // single op
}

//===================================================================================================================== Elementwise Tensor Multiplicatio

// Define a struct that implements the Element_Operation interface
type EWMultiplication struct{}

func (es EWMultiplication) ExecuteElementwiseOp(a, b float64) float64 {
	return a * b
}

func (ba EWMultiplication) Execute(tensors ...*Tensor) *Tensor {
	A, B := tensors[0], tensors[1]
	return ElementwiseOp(A, B, EWMultiplication{})
}

// This function performs elementwise addition on two tensors with optional batching
func Multiply(A *Tensor, B *Tensor, batching bool) *Tensor {

	ewMultiplication := EWMultiplication{} // Create an instance of EWMultiplication

	if batching {
		return BatchedOperation(ewMultiplication, A, B) // batched op
	}
	return ewMultiplication.Execute(A, B) // single op
}
