package GLA

// axis_ops.go contains functions that perform operations along a specified
// axis of a tensor that result in the axis being collapsed due to the nature
// of the operation. All functions in this source file currently rely on the
// Axis_ElementOperation() function to perform this type of axis operation.

// Currently this includes: Sum_Axis(), Mean_Axis(), Var_Axis(), Std_Axis()

import (
	//"fmt"
	"math"
)

// Operation is an interface representing an operation on individual tensor elements
type Element_Operation interface {
	Apply_ElementOp(float64, float64) float64 // performs op on 2 float64s, returns float64
}

// Axis_ElementOperation applies an operation to each element along a given axis of the tensor. This results in a Tensor that is 1 Dim
// smaller because that axis has beem collapsed in the process. This function is useful for generalizing operations of this type.
func (A *Tensor) Axis_ElementOperation(axis int, op Element_Operation) *Tensor {
	if axis < 0 || axis >= len(A.Shape) {
		panic("Withing Axis_ElementOperation() --- Invalid axis")
	}

	// Calculate new shape and initialize data for the result tensor with the axis removed
	newShape := make([]int, len(A.Shape)-1)
	copy(newShape, A.Shape[:axis])          // <--- copy everything before the axis
	copy(newShape[axis:], A.Shape[axis+1:]) // <--- copy everything after the axis
	newData := make([]float64, Product(newShape))
	indices := make([]int, len(A.Shape))

	// Perform the operation along the specified axis.
	for i := 0; i < len(A.Data); i++ {

		// compute 1D index of the result tensor
		concatIndices := append(indices[:axis], indices[axis+1:]...)
		resultIndex := Index(concatIndices, newShape) // <--- compute the 1D index of the result tensor

		// Apply operation at the current index.
		newData[resultIndex] = op.Apply_ElementOp(newData[resultIndex], A.Data[i])

		// Increment multi-dimensional indices.
		for dim := len(A.Shape) - 1; dim >= 0; dim-- {
			indices[dim]++
			if indices[dim] < A.Shape[dim] {
				break
			}
			indices[dim] = 0
		}
	}

	return &Tensor{Shape: newShape, Data: newData}
}

//------------------------------------------------------------------------------------------------------ Sum_Axis()

// SumOperation defines summation on tensor elements.
type SumOperation struct{}

func (s SumOperation) Apply_ElementOp(a, b float64) float64 {
	return a + b
}

// SumAxisBatchOperation performs batched summation along an axis.
type SumAxisBatchOperation struct{ axis int }

func (op SumAxisBatchOperation) Execute(tensor *Tensor) *Tensor {
	return tensor.Axis_ElementOperation(op.axis, SumOperation{})
}

// Sum_Axis sums tensor elements along a specified axis.
func (A *Tensor) Sum_Axis(axis int, batching bool) *Tensor {
	if !batching {
		return A.Axis_ElementOperation(axis, SumOperation{})
	} else {
		batchOp := SumAxisBatchOperation{axis: axis}
		return Batch_Tensor_Tensor_Operation(batchOp, A)
	}
}

//------------------------------------------------------------------------------------------------------ Mean_Axis()

// Mean_Axis calculates the mean of tensor elements along a specified axis.
func (A *Tensor) Mean_Axis(axis int, batching bool) *Tensor {
	sumTensor := A.Sum_Axis(axis, batching)
	count := A.Shape[axis]
	for i := range sumTensor.Data {
		sumTensor.Data[i] /= float64(count)
	}
	return sumTensor
}

//------------------------------------------------------------------------------------------------------ Var_Axis()

// VarOperation calculates variance.
type VarOperation struct {
	mean float64
}

func (v VarOperation) Apply_ElementOp(a, b float64) float64 {
	diff := b - v.mean
	return a + diff*diff
}

// VarAxisBatchOperation performs batched variance calculation along an axis.
type VarAxisBatchOperation struct{ axis int }

func (op VarAxisBatchOperation) Execute(tensor *Tensor) *Tensor {
	meanTensor := tensor.Mean_Axis(op.axis, false)
	varOp := VarOperation{mean: meanTensor.Data[0]}
	return tensor.Axis_ElementOperation(op.axis, varOp)
}

// Var_Axis calculates variance along a specified axis.
func (A *Tensor) Var_Axis(axis int, batching bool) *Tensor {
	batchOp := VarAxisBatchOperation{axis: axis}
	if !batching {
		return batchOp.Execute(A)
	} else {
		return Batch_Tensor_Tensor_Operation(batchOp, A)
	}
}

//------------------------------------------------------------------------------------------------------ Std_Axis()

// StdAxisBatchOperation performs batched standard deviation calculation.
type StdAxisBatchOperation struct{ axis int }

func (op StdAxisBatchOperation) Execute(tensor *Tensor) *Tensor {
	varTensor := tensor.Var_Axis(op.axis, false)
	for i := range varTensor.Data {
		varTensor.Data[i] = math.Sqrt(varTensor.Data[i])
	}
	return varTensor
}

// Std_Axis calculates standard deviation along a specified axis.
func (A *Tensor) Std_Axis(axis int, batching bool) *Tensor {
	batchOp := StdAxisBatchOperation{axis: axis}
	if !batching {
		return batchOp.Execute(A)
	} else {
		return Batch_Tensor_Tensor_Operation(batchOp, A)
	}
}
