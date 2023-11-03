package main

// This source file contians Tensor operations performed over a specified axis

import (
	"math"
)

// =========================================================================================================== Axis Operations Generalization

// Operation is an interface representing an operation on individual tensor elements
type Operation interface {
	Apply(float64, float64) float64 // performs op on 2 float64s, returns float64
}

// AxisOperation applies a specified operation along a given axis of the tensor.
//
// This function performs computations along a specific axis of the tensor, collapsing
// the tensor along that axis based on the provided operation. The result of this
// operation is a new tensor with one fewer dimension than the original tensor.
// For every position along the specified axis, there is a unique combination of indices
// for all other axes. The function collapses the tensor along the specified axis by
// performing the specified operation on the values at each unique combination of indices
// for the other axes, resulting in a new tensor where the dimension along the specified
// axis is removed.
//
// The function takes two parameters:
//   - axis: an integer that specifies the axis along which the operation is performed.
//   - operation: an Operation interface that defines the specific operation to be applied.
//     The operation is applied to pairs of float64 values and returns a float64 result.
//
// It returns a pointer to a new Tensor that is the result of applying the operation.
func (A *Tensor) AxisOperation(axis int, op Operation) *Tensor {
	if axis < 0 || axis >= len(A.shape) { // <--- Check that the axis is valid.
		panic("Invalid axis")
	}

	// Calculate the shape of the result tensor.
	newShape := make([]int, len(A.shape)-1) // <--- set new shape 1 dim smaller than original
	copy(newShape, A.shape[:axis])          // <--- Remove specified dimension by excluding it from the copy.
	copy(newShape[axis:], A.shape[axis+1:]) // <---

	// Initialize the data for the result tensor.
	newData := make([]float64, Product(newShape))
	indices := make([]int, len(A.shape)) // <--- multi dim indexing for iterating through tensor

	// Perform the operation along the specified axis.
	for i := 0; i < len(A.data); i++ {
		// Concatenate the indices before and after the specified axis to form the reduced-dimension indices.
		concatIndices := append(indices[:axis], indices[axis+1:]...)

		// Convert the reduced-dimension indices to a flattened index for the result tensor.
		resultIndex := Index(concatIndices, newShape)

		// Apply the operation to the current element in the tensor and update the result tensor.
		newData[resultIndex] = op.Apply(newData[resultIndex], A.data[i])

		// Increment the multi-dimensional indices, like an odometer with each wheel representing a dimension's index.
		// For ex: in a 3x3 matrix this would be [0,0] -> [0,1] -> [0,2] -> [1,0] -> [1,1] -> [1,2] -> [2,0] -> [2,1] -> [2,2]
		for dim := len(A.shape) - 1; dim >= 0; dim-- {
			indices[dim]++ // Advance the index in the current dimension.

			if indices[dim] < A.shape[dim] {
				// If the index within the current dimension is still within bounds, continue with the next element.
				break
			}
			// If the current dimension's index "overflows", reset it to zero and move to increment the next higher dimension.
			indices[dim] = 0
		}
	}

	return &Tensor{shape: newShape, data: newData}
}

// ============================================================================================================ Summation on an Axis

// SumOperation represents a summation operation.
type SumOperation struct{}

func (s SumOperation) Apply(a, b float64) float64 { // Apply summation on two float64 values.
	return a + b
}
func (A *Tensor) Sum(axis int) *Tensor {
	return A.AxisOperation(axis, SumOperation{}) // sum along an axis
}

// ============================================================================================================ Mean on an Axis

// Mean calculates the mean of elements in a tensor along a specified axis
func (A *Tensor) Mean(axis int) *Tensor {
	sumTensor := A.Sum(axis) // sum along axis
	count := A.shape[axis]
	for i := range sumTensor.data {
		sumTensor.data[i] /= float64(count)
	}
	return sumTensor
}

// ============================================================================================================ Variance on an Axis

// VarOperation represents a variance calculation operation.
type VarOperation struct {
	mean float64 // used within Apply() to calculate variance
}

func (v VarOperation) Apply(a, b float64) float64 { // apply variance calculation on two float64 values.
	/// variance = sum((x - mean)^2) / n
	diff := b - v.mean
	return a + diff*diff
}
func (A *Tensor) Var(axis int) *Tensor {
	meanTensor := A.Mean(axis)                      // mean along axis
	varOp := VarOperation{mean: meanTensor.data[0]} // pass the mean to the operation
	return A.AxisOperation(axis, varOp)             // variance along an axis
}

// ============================================================================================================ Standard Deviation on an Axis

// Std() calculates the standard deviation of elements in a tensor along a specified axis.
func (A *Tensor) Std(axis int) *Tensor {
	varTensor := A.Var(axis)
	for i := range varTensor.data {
		varTensor.data[i] = math.Sqrt(varTensor.data[i])
	}
	return varTensor
}
