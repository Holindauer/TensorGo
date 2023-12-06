package TG

import "fmt"

// all_elements.go contains operations performed over all elements of a tensor in which the the result is a
// version of the original Tensor with the same shape. This is useful for operations such as normalization an
// elementwise multiplication.

// Currently this includes the follwing functions: Add(), Subtract(), Scalar_Mult_(), Normalize()

// =========================================================================================================== Elementwise Tensor Operations gereralization

// This interace is used to generalize elementwise tensor operations on the level of individual elements
type Element_Operation interface {
	Element_Op(a, b float64) float64
}

// This function is a generalization of elementwise tensor operations. It takes in two tensors and an Element_Operation.
func Elementwise_Operation(A *Tensor, B *Tensor, op Element_Operation) *Tensor {

	if !Same_Shape(A, B) {
		fmt.Println("Within Elementwise_Operation(): -- A.Shape:", A.Shape, "B.Shape:", B.Shape)
		panic("Within Elementwise_Operation(): Tensors must have the same shape")
	}

	C := Zero_Tensor(A.Shape, false)
	for i := 0; i < len(A.Data); i++ {
		C.Data[i] = op.Element_Op(A.Data[i], B.Data[i]) // perform operation with elements
	}
	return C // <-- pointer
}

//=========================================================================================================== Elementwise Tensor Addition

// Define a struct that implements the Element_Operation interface
type Elementwise_Addition struct{}

func (ea Elementwise_Addition) Element_Op(a, b float64) float64 {
	return a + b
}

// define a struct that implements the Batch_TwoTensor_Tensor_Interface interface. See batching.go
type Batched_Addition struct{}

func (ba Batched_Addition) Execute(A, B *Tensor) *Tensor {
	return Elementwise_Operation(A, B, Elementwise_Addition{})
}

// This function performs elementwise addition on two tensors with optional batching
func Add(A *Tensor, B *Tensor, batching bool) *Tensor {

	if batching {
		return Batch_TwoTensor_Tensor_Operation(Batched_Addition{}, A, B) // batched op
	}
	return Batched_Addition{}.Execute(A, B) // single op
}

//=========================================================================================================== Elementwise Tensor Subtraction

// Define a struct that implements the Element_Operation interface
type Elementwise_Subtraction struct{}

func (es Elementwise_Subtraction) Element_Op(a, b float64) float64 {
	return a - b
}

// define a struct that implements the Batch_TwoTensor_Tensor_Interface interface. See batching.go
type Batched_Subtraction struct{}

func (ba Batched_Subtraction) Execute(A, B *Tensor) *Tensor {
	return Elementwise_Operation(A, B, Elementwise_Subtraction{})
}

// This function performs elementwise addition on two tensors with optional batching
func Subtract(A *Tensor, B *Tensor, batching bool) *Tensor {

	if batching {
		return Batch_TwoTensor_Tensor_Operation(Batched_Subtraction{}, A, B) // batched op
	}
	return Batched_Subtraction{}.Execute(A, B) // single op
}

//=========================================================================================================== Elementwise Tensor Multiplicatio

// Define a struct that implements the Element_Operation interface
type Elementwise_Multiplication struct{}

func (es Elementwise_Multiplication) Element_Op(a, b float64) float64 {
	return a * b
}

// define a struct that implements the Batch_TwoTensor_Tensor_Interface interface. See batching.go
type Batched_Multiplication struct{}

func (ba Batched_Multiplication) Execute(A, B *Tensor) *Tensor {
	return Elementwise_Operation(A, B, Elementwise_Multiplication{})
}

// This function performs elementwise addition on two tensors with optional batching
func Multiply(A *Tensor, B *Tensor, batching bool) *Tensor {

	if batching {
		return Batch_TwoTensor_Tensor_Operation(Batched_Multiplication{}, A, B) // batched op
	}
	return Batched_Multiplication{}.Execute(A, B) // single op
}

// =========================================================================================================== Scalar Multiplication

// Define a struct that implements the Batch_Tensor_Tensor_Operation interface. See batching.go for more details
type Batched_Scalar_Multiplication struct{ scalar float64 }

func (bsm Batched_Scalar_Multiplication) Execute(A *Tensor) *Tensor {
	// create new tensor to store result
	cA := A.Copy()
	for i := 0; i < len(A.Data); i++ {
		cA.Data[i] *= bsm.scalar
	}
	return cA
}

// This funciton performs scalar multiplication on a tensor in place
// It returns a pointer to a new tensor
func (A *Tensor) Scalar_Mult(scalar float64, batching bool) *Tensor {

	if batching {
		return Batch_Tensor_Tensor_Operation(Batched_Scalar_Multiplication{scalar: scalar}, A) // batched op
	}
	return Batched_Scalar_Multiplication{scalar: scalar}.Execute(A) // single op
}

// =========================================================================================================== Normalilze Tensor Across All Elements

type Batched_Normalize struct{}

func (bn Batched_Normalize) Execute(A *Tensor) *Tensor {
	// convert A to a vector Tensor in order to use Norm()
	A_vector := A.Copy().Reshape([]int{Product(A.Shape)}, false)
	A_Norm := Norm(A_vector, false)

	// iterate over all elements of A and divide by A_Norm
	for i := 0; i < len(A.Data); i++ {
		A.Data[i] /= A_Norm.Data[0] // <-- single element tensor
	}
	return A
}

// Normalize divides each element of a tensor by the tensor's norm. This happens in place There is optional batching
func (A *Tensor) Normalize(batching bool) *Tensor {

	if batching {
		return Batch_Tensor_Tensor_Operation(Batched_Normalize{}, A) // batched op
	}
	return Batched_Normalize{}.Execute(A) // single op
}
