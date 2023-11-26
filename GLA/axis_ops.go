package GLA

import (
	//"fmt"
	"math"
)

// Operation is an interface representing an operation on individual tensor elements
type Operation interface {
	Apply(float64, float64) float64 // performs op on 2 float64s, returns float64
}

// AxisOperation applies an operation along a given axis of the tensor.
func (A *Tensor) AxisOperation(axis int, op Operation) *Tensor {
	if axis < 0 || axis >= len(A.Shape) {
		panic("Invalid axis")
	}

	// Calculate new shape and initialize data for the result tensor.
	newShape := make([]int, len(A.Shape)-1)
	copy(newShape, A.Shape[:axis])
	copy(newShape[axis:], A.Shape[axis+1:])
	newData := make([]float64, Product(newShape))
	indices := make([]int, len(A.Shape))

	// Perform the operation along the specified axis.
	for i := 0; i < len(A.Data); i++ {
		concatIndices := append(indices[:axis], indices[axis+1:]...)
		resultIndex := Index(concatIndices, newShape)
		newData[resultIndex] = op.Apply(newData[resultIndex], A.Data[i])

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

// SumOperation defines summation on tensor elements.
type SumOperation struct{}

func (s SumOperation) Apply(a, b float64) float64 {
	return a + b
}

// SumAxisBatchOperation performs batched summation along an axis.
type SumAxisBatchOperation struct{ axis int }

func (op SumAxisBatchOperation) Execute(tensor *Tensor) *Tensor {
	return tensor.AxisOperation(op.axis, SumOperation{})
}

// Sum_Axis sums tensor elements along a specified axis.
func (A *Tensor) Sum_Axis(axis int, batching bool) *Tensor {
	if !batching {
		return A.AxisOperation(axis, SumOperation{})
	} else {
		batchOp := SumAxisBatchOperation{axis: axis}
		return Batch_Tensor_Tensor_Operation(batchOp, A)
	}
}

// Mean_Axis calculates the mean of tensor elements along a specified axis.
func (A *Tensor) Mean_Axis(axis int, batching bool) *Tensor {
	sumTensor := A.Sum_Axis(axis, batching)
	count := A.Shape[axis]
	for i := range sumTensor.Data {
		sumTensor.Data[i] /= float64(count)
	}
	return sumTensor
}

// VarOperation calculates variance.
type VarOperation struct {
	mean float64
}

func (v VarOperation) Apply(a, b float64) float64 {
	diff := b - v.mean
	return a + diff*diff
}

// VarAxisBatchOperation performs batched variance calculation along an axis.
type VarAxisBatchOperation struct{ axis int }

func (op VarAxisBatchOperation) Execute(tensor *Tensor) *Tensor {
	meanTensor := tensor.Mean_Axis(op.axis, false)
	varOp := VarOperation{mean: meanTensor.Data[0]}
	return tensor.AxisOperation(op.axis, varOp)
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
