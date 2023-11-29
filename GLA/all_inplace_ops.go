package GLA

// all_elements.go contains operations performed over all elements of a tensor in which the the result is a
// version of the original Tensor with the same shape. This is useful for operations such as normalization an
// elementwise multiplication.

// Currently this includes the follwing functions: Add(), Subtract(), Scalar_Mult_(), Normalize()

//=========================================================================================================== Elementwise Tensor Addition

// This function performs elementwise addition on two tensors
// The tensors must have the same shape. It returns a pointer to a new tensor
func Add(A *Tensor, B *Tensor) *Tensor {

	if !Same_Shape(A, B) {
		panic("Within Add(): Tensors must have the same shape")
	}

	C := Zero_Tensor(A.Shape, false)
	for i := 0; i < len(A.Data); i++ {
		C.Data[i] = A.Data[i] + B.Data[i] // add elements
	}

	return C // <-- pointer
}

//=========================================================================================================== Elementwise Tensor Subtraction

// This function performs elementwise subtraction on two tensors
// The tensors must have the same shape. It returns a pointer to a new tensor
func Subtract(A *Tensor, B *Tensor) *Tensor {

	// Check that the tensors have the same shape
	if !Same_Shape(A, B) {
		panic("Within Subtract(): Tensors must have the same shape")
	}

	C := Zero_Tensor(A.Shape, false)
	for i := 0; i < len(A.Data); i++ {
		C.Data[i] = A.Data[i] - B.Data[i] // subtract elements
	}
	return C
}

// =========================================================================================================== Elementwise Scalar Multiplication

// This funciton performs scalar multiplication on a tensor in place
// It returns a pointer to a new tensor
func (A *Tensor) Scalar_Mult_(scalar float64) *Tensor {

	// create new tensor to store result
	cA := Zero_Tensor(A.Shape, false)
	for i := 0; i < len(A.Data); i++ {
		cA.Data[i] *= scalar
	}
	return cA
}

// =========================================================================================================== Normalilze Tensor Across All Elements

type Batched_Normalize struct{}

func (bn Batched_Normalize) Execute(A *Tensor) *Tensor {
	// convert A to a vector Tensor in order to use Norm()
	A_vector := A.Copy().Reshape([]int{Product(A.Shape)})
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
