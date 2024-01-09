package TG

// all_elements.go contains operations performed over all elements of a tensor in which the the result is a
// version of the original Tensor with the same shape. This is useful for operations such as normalization an
// elementwise multiplication.

// Currently this includes the follwing functions: Add(), Subtract(), Scalar_Mult_(), Normalize()

// =========================================================================================================== Scalar Multiplication

// Define a struct that implements the Batch_Tensor_Tensor_Operation interface. See batching.go for more details
type ScalarMultOp struct{ scalar float64 }

func (bsm ScalarMultOp) Execute(tensors ...*Tensor) *Tensor {

	A := tensors[0]

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

	// initialize the batched op
	scalarMult := ScalarMultOp{scalar: scalar}

	if batching {
		return BatchedOperation(scalarMult, A) // batched op
	}
	return scalarMult.Execute(A) // single op
}

// =========================================================================================================== Normalilze Tensor Across All Elements

type NormalizizeOp struct{}

func (bn NormalizizeOp) Execute(tensors ...*Tensor) *Tensor {

	A := tensors[0]

	// convert A to a vector Tensor in order to use Norm()
	A_vector := A.Copy().Reshape([]int{Product(A.Shape)}, false)
	A_Norm := A_vector.Norm(false)

	// iterate over all elements of A and divide by A_Norm
	for i := 0; i < len(A.Data); i++ {
		A.Data[i] /= A_Norm.Data[0] // <-- single element tensor
	}
	return A
}

// Normalize divides each element of a tensor by the tensor's norm. This happens in place There is optional batching
func (A *Tensor) Normalize(batching bool) *Tensor {

	// initialize the batched op
	normalize := NormalizizeOp{}

	if batching {
		return BatchedOperation(normalize, A) // batched op
	}
	return normalize.Execute(A) // single op
}

// =========================================================================================================== Standardize a Batched Tensor

// This struct contains the means and std's along each feature. It is used to define the below method that standardizes each batch element.
type StandardizeOp struct{ A_Mean_Axis_0, A_Std_Axis_0 *Tensor }

// By treating standardizardization as a batched operation, we can ensure that the Tensors containing the mean and std line up
// element by element with the element of the Tensor that is being standardized. This is because the axis op collapsed the 0'th axis
func (s StandardizeOp) Execute(tensors ...*Tensor) *Tensor {

	A := tensors[0]

	Standardized_A := Zero_Tensor(A.Shape, false)
	indices := make([]int, len(A.Shape)) // <--- to hold a single multi-dimensional indices

	// Consider a 3x3x3 tensor. The indices will start at [0, 0, 0], [0, 0, 1], then [0, 0, 2], [0, 1, 0]... etc.
	for i := 0; i < len(A.Data); i++ {
		// Standarize the current index.
		resultIndex := A.Index(indices) // <--- compute the 1D index of the result tensor

		if s.A_Std_Axis_0.Data[indices[0]] == 0 {
			Standardized_A.Data[resultIndex] = 0 // Handle the case where the standard deviation is zero.
		} else {
			Standardized_A.Data[resultIndex] = (A.Data[resultIndex] - s.A_Mean_Axis_0.Data[indices[0]]) / s.A_Std_Axis_0.Data[indices[0]]
		}

		// Drecrement multi-dimensional indices.
		for dim := len(A.Shape) - 1; dim >= 0; dim-- {
			indices[dim]++
			if indices[dim] < A.Shape[dim] { // break to the next iter if we havemt reached the end of the current dimension
				break
			}
			indices[dim] = 0
		}
	}

	return Standardized_A
}

// Standardize() is a batched-only method that standardizes each element of a batched tensors using z-score normalization z = (x - mean) / std
// where x is the element of the batched tensor, and mean and std are the mean and standard deviation of that specific
// element in the batched tensor.

// Standardization() works for batched Tensors of arbitrary dimmensionality by first Using Mean_Axis() and Std_Axis() operations
// to collapse the batch dimmension of the Tensor into their respective statistics. Then, each element is passed to the Execute()
// Method of the Standardize_Operation struct, which iterates through the mulit-dimensional indices of the Tensor standarizing each
func (A *Tensor) Standardize() *Tensor {

	// In order for this to work we need to stack the batch
	A_Mean_Axis_0 := A.Mean_Axis(0, false) // <--- batching set to false because we want to mean each feature
	A_Std_Axis_0 := A.Std_Axis(0, false)

	op := StandardizeOp{A_Mean_Axis_0, A_Std_Axis_0}
	return BatchedOperation(op, A)
}
