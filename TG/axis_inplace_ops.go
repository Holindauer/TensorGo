package TG

// axis_ops.go contains functions that perform an operation along a specified axis of a Tensor that keeps the dimmensions of the Tensor in tact.

// Currently this includes: Normalize_Axis()

//"fmt"

//------------------------------------------------------------------------------------------------------ Axis_ElementOperation()

// Axis_ElementOperation applies an operation to each element along a given axis of a Tensor. This results in a Tensor that
// is the same shape as the original Tensor. Due to the nature of this problem, we can tale advantage of the code that was
// written for performing batched operations. By Transposing the axis of operation to the first axis, the operation can be
// treated as a batched operation. To which the result can be Transposed back to the original shape.

type Inplace_Operation interface {
	Apply_InplaceOp(*Tensor)
}

// BatchInplaceOperation applies an Inplace_Operation to a slice of the tensor
type BatchInplaceOperation struct {
	op Inplace_Operation
}

func (b *BatchInplaceOperation) Execute(tensors ...*Tensor) *Tensor {

	A := tensors[0]

	// Apply the Inplace_Operation to the tensor
	b.op.Apply_InplaceOp(A)
	return A
}

func (A *Tensor) Axis_InplaceOperation(axis int, op Inplace_Operation) *Tensor {
	if axis < 0 || axis >= len(A.Shape) {
		panic("Within Axis_InplaceOperation() --- Invalid axis")
	}

	// Transpose Tensor so that the axis of operation is the first axis
	axisReordering := Permute_Shape(A.Shape, axis, 0) // Using the previously defined function
	A_Transpose := A.Permute(axisReordering)

	// Define the batch operation that applies op
	batchOp := &BatchInplaceOperation{op: op} // BatchInplaceOperation needs to be defined

	// Perform the operation along the first axis using Batch_Tensor_Tensor_Operation
	result := BatchedOperation(batchOp, A_Transpose)

	// Transpose the result back to the original shape
	resultTranspose := result.Permute(Permute_Shape(result.Shape, 0, axis))

	return resultTranspose
}

//------------------------------------------------------------------------------------------------------ Normalize_Axis()

// NormalizeOperation implements the normalization operation for a tensor slice
type NormalizeOperation struct{}

func (nop NormalizeOperation) Apply_InplaceOp(tensorSlice *Tensor) {
	norm := calculateNorm(tensorSlice) // Implement this function to calculate the norm of the tensor slice
	for i := range tensorSlice.Data {
		tensorSlice.Data[i] /= norm
	}
}

// Normalize_Axis normalizes a tensor along the specified axis
func (A *Tensor) Normalize_Axis(axis int) *Tensor {
	return A.Axis_InplaceOperation(axis, NormalizeOperation{})
}

// calculateNorm calculates the norm of a tensor slice
func calculateNorm(tensorSlice *Tensor) float64 {
	return tensorSlice.Norm(false).Data[0]
}
