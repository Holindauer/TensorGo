package main

// this source file contatins statistical functions for tensors
// The file is broken up into functions that perform statistical operations on a single specified axis
// vs functions that perform statistical operations on all elements of the tensor at the same time

// Functions/Methods for computing on all elements at once are denoted by the _All suffix

import (
	"math"
	"sync"
)

//--------------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------Operations on a Single Axis-------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------

// =========================================================================================================== Axis Operations Generalization
// Operation is an interface representing an operation on a tensor along a specific axis.
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
func (t *Tensor) AxisOperation(axis int, op Operation) *Tensor {
	if axis < 0 || axis >= len(t.shape) { // Check that the axis is valid.
		panic("Invalid axis")
	}

	// Calculate the shape of the result tensor.
	newShape := make([]int, len(t.shape)-1)
	copy(newShape, t.shape[:axis]) // Remove specified dimension by excluding it from the copy.
	copy(newShape[axis:], t.shape[axis+1:])

	// Initialize the data for the result tensor.
	newData := make([]float64, Product(newShape))
	indices := make([]int, len(t.shape))

	// Perform the operation along the specified axis.
	for i := 0; i < len(t.data); i++ {
		concatIndices := append(indices[:axis], indices[axis+1:]...) // Remove the specified axis by appending the indices before and after it.
		resultIndex := Index(concatIndices, newShape)
		newData[resultIndex] = op.Apply(newData[resultIndex], t.data[i]) // Apply the operation to the current value and the result so far.

		// Increment the multi-dimensional indices.
		for dim := len(t.shape) - 1; dim >= 0; dim-- {
			indices[dim]++
			if indices[dim] < t.shape[dim] {
				break
			}
			indices[dim] = 0
		}
	}
	return &Tensor{shape: newShape, data: newData}
}

// Helper function for AxisOperation() for computing the product of elements in a slice
func Product(shape []int) int {
	product := 1
	for _, dim := range shape {
		product *= dim
	}
	return product
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

//--------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------Operations on All Elements at once--------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------

//=========================================================================================================== Operations on All Elements Generalization

// AllOperation is an interface representing an operation applied to all elements of a tensor.
type AllOperation interface {
	Apply(*Tensor, int, int) float64  // Apply performs the operation on a chunk of the tensor's data.
	CombineResults([]float64) float64 // CombineResults combines the results from all chunks.
}

// SumAllOperation represents a summation operation over the entire tensor.
type SumAllOperation struct{}

// Apply performs the summation on a chunk of the tensor's data for a go routine.
func (s SumAllOperation) Apply(t *Tensor, start, end int) float64 {
	var sum float64
	for i := start; i < end; i++ {
		sum += t.data[i]
	}
	return sum
}

// CombineResults combines the summation results from all chunks of a go routine.
func (s SumAllOperation) CombineResults(results []float64) float64 {
	var sum float64
	for _, v := range results {
		sum += v
	}
	return sum
}

// MeanAllOperation represents a mean calculation operation over the entire tensor.
type MeanAllOperation struct{}

// Apply performs the mean calculation on a chunk of the tensor's data for a go routine.
func (m MeanAllOperation) Apply(t *Tensor, start, end int) float64 {
	sumOp := SumAllOperation{}
	sum := sumOp.Apply(t, start, end)
	return sum / float64(end-start)
}

// CombineResults combines the mean results from all chunks of a go routine.
func (m MeanAllOperation) CombineResults(results []float64) float64 {
	sumOp := SumAllOperation{}
	sum := sumOp.CombineResults(results)
	return sum / float64(len(results))
}

// VarAllOperation represents a variance calculation operation over the entire tensor.
type VarAllOperation struct {
	mean float64
}

// Apply performs the variance calculation on a chunk of the tensor's data.
func (v VarAllOperation) Apply(t *Tensor, start, end int) float64 {
	var variance float64
	for i := start; i < end; i++ {
		diff := t.data[i] - v.mean
		variance += diff * diff
	}
	return variance
}

// CombineResults combines the variance results from all chunks.
func (v VarAllOperation) CombineResults(results []float64) float64 {
	sumOp := SumAllOperation{}
	sum := sumOp.CombineResults(results)
	return sum / float64(len(results))
}

// AllOperation applies a specified operation to all elements of the tensor.
//
// This function performs computations on all elements of the tensor, utilizing concurrency to
// speed up the process. The results from all chunks of data processed concurrently are then
// combined to produce the final result of the operation.
//
// The function takes one parameter:
//   - op: an AllOperation interface that defines the specific operation to be applied.
//     The operation is applied to chunks of the tensor's data and results are combined to produce a final result.
//
// It returns a float64 that is the result of applying the operation.
func (t *Tensor) AllOperation(op AllOperation) float64 {
	var wg sync.WaitGroup
	var mutex = &sync.Mutex{}

	numGoroutines := 4
	chunkSize := len(t.data) / numGoroutines
	results := make([]float64, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)

		start := i * chunkSize
		end := start + chunkSize

		if i == numGoroutines-1 {
			end = len(t.data) // Ensure the last chunk includes any remaining elements
		}

		go func(i int, start, end int) {
			defer wg.Done()

			chunkResult := op.Apply(t, start, end) // Apply the operation to the chunk of data.

			mutex.Lock()
			results[i] = chunkResult
			mutex.Unlock()
		}(i, start, end)
	}
	wg.Wait()

	return op.CombineResults(results) // Combine the results from all chunks.
}

func (t *Tensor) Sum_All() float64 {
	sumOp := SumAllOperation{}
	return t.AllOperation(sumOp)
}

func (t *Tensor) Mean_All() float64 {
	meanOp := MeanAllOperation{}
	return t.AllOperation(meanOp)
}

func (t *Tensor) Var_All() float64 {
	mean := t.Mean_All()
	varOp := VarAllOperation{mean: mean}
	return t.AllOperation(varOp)
}

func (t *Tensor) Std_All() float64 {
	varOp := VarAllOperation{mean: t.Mean_All()}
	variance := t.AllOperation(varOp)
	return math.Sqrt(variance)
}
