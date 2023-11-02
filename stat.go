package main

// this source file contatins statistical functions for tensors
// The file is broken up into functions that perform statistical operations on a single specified axis
// vs functions that perform statistical operations on all elements of the tensor at the same time

// Functions/Methods for computing on all elements at once are denoted by the _All suffix

import (
	"math"
	"sync"
)

//------------------------------------------------------------------------------------------------------------Operations on Axis
// Operation is an interface representing an operation on a tensor along a specific axis.

type Operation interface {
	Apply(float64, float64) float64 // performs op on 2 float64s, returns float64
}

// SumOperation represents a summation operation.
type SumOperation struct{}

func (s SumOperation) Apply(a, b float64) float64 { // Apply summation on two float64 values.
	return a + b
}

// VarOperation represents a variance calculation operation.
type VarOperation struct {
	mean float64 // used within Apply() to calculate variance
}

func (v VarOperation) Apply(a, b float64) float64 { // apply variance calculation on two float64 values.
	/// variance = sum((x - mean)^2) / n
	diff := b - v.mean
	return a + diff*diff
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

// Sum performs a summation along a specified axis of the tensor.
func (A *Tensor) Sum(axis int) *Tensor {
	return A.AxisOperation(axis, SumOperation{})
}

// Var calculates the variance along a specified axis of the tensor.
func (A *Tensor) Var(axis int) *Tensor {
	meanTensor := A.Mean(axis)
	varOp := VarOperation{mean: meanTensor.data[0]} // Assuming meanTensor is a scalar for simplicity.
	return A.AxisOperation(axis, varOp)
}

// Mean calculates the mean of elements in a tensor along a specified axis
func (A *Tensor) Mean(axis int) *Tensor {
	sumTensor := A.Sum(axis) // sum along axis
	count := A.shape[axis]
	for i := range sumTensor.data {
		sumTensor.data[i] /= float64(count)
	}
	return sumTensor
}

// Std() calculates the standard deviation of elements in a tensor along a specified axis.
func (A *Tensor) Std(axis int) *Tensor {
	varTensor := A.Var(axis)
	for i := range varTensor.data {
		varTensor.data[i] = math.Sqrt(varTensor.data[i])
	}
	return varTensor
}

//------------------------------------------------------------------------------------------------------------Operations on All Elements at once

// this function sums all elements of the contiguous data in tensor.data
// it utilizes concurrency and mutexes to speed up the process. This function
// will be used within the Var() and Std() functions.
func (A *Tensor) Sum_All() float64 {
	var sum float64
	var wg sync.WaitGroup
	var mutex = &sync.Mutex{}

	numGoroutines := 4
	chunkSize := len(A.data) / numGoroutines

	for i := 0; i < numGoroutines; i++ {

		wg.Add(1) // Increment the WaitGroup counter

		start := i * chunkSize //  compute bounds of the chunk
		end := start + chunkSize

		if i == numGoroutines-1 {
			end = len(A.data) // Ensure the last chunk includes any remaining elements
		}

		go computeSum(A, start, end, &sum, mutex, &wg)
	}
	wg.Wait()
	return sum
}

// This is a helper function for Sum_All() above. It computes the sum of a chunk of the vectors
func computeSum(A *Tensor, start int, end int, sum *float64, mutex *sync.Mutex, wg *sync.WaitGroup) {
	defer wg.Done()

	for i := start; i < end; i++ {
		mutex.Lock()
		*sum += A.data[i]
		mutex.Unlock()
	}
}

// Mean_All() calculates the mean of all elements in a tensor. This operation
// results in a tensor with one fewer dimension than the original tensor. It utilizes .Sum_All()
func (A *Tensor) Mean_All() float64 {
	sum := A.Sum_All()
	return sum / float64(len(A.data))
}

// Var_All() calculates the variance of all elements in a tensor.
// Variance is define as the average of the squared differences from the mean
// it utilizes .Mean_All() and Concurrency w/ mutex for speed
func (A *Tensor) Var_All() float64 {
	mean := A.Mean_All()
	variance := 0.0
	var wg sync.WaitGroup
	var mutex = &sync.Mutex{}

	numGoroutines := 4
	chunkSize := len(A.data) / numGoroutines

	for i := 0; i < numGoroutines; i++ {

		wg.Add(1) // Increment the WaitGroup counter

		start := i * chunkSize //  compute bounds of the chunk
		end := start + chunkSize

		if i == numGoroutines-1 {
			end = len(A.data) // Ensure the last chunk includes any remaining elements
		}

		go computeVar(A, start, end, mean, &variance, mutex, &wg)
	}

	wg.Wait()
	return variance / float64(len(A.data)) // <-- divide by number of elements to get average
}

// This is a helper function for Var_All() above. It computes the variance of a chunk of the vectors
func computeVar(A *Tensor, start int, end int, mean float64, variance *float64, mutex *sync.Mutex, wg *sync.WaitGroup) {
	defer wg.Done()

	for i := start; i < end; i++ {
		mutex.Lock()
		*variance += (A.data[i] - mean) * (A.data[i] - mean) // <-- this is the variance formula
		mutex.Unlock()
	}
}

// Std_All()
