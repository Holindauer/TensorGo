package main

// this source file contatins statistical functions for tensors
// The file is broken up into functions that perform statistical operations on a single specified axis
// vs functions that perform statistical operations on all elements of the tensor at the same time

// Functions/Methods for computing on all elements at once are denoted by the _All suffix

import (
	"sync"
)

//------------------------------------------------------------------------------------------------------------Operations on Axis

// Sum calculates the sum of elements in a tensor along a specified axis. This operation
// results in a tensor with one fewer dimension than the original tensor. For each position
// along the specified axis, there exists a unique combination of indices for all other axes.
// The function collapses the tensor by summing the values at each unique combination of
// indices for the other axes, resulting in a new tensor where the dimension along the
// specified axis is removed.
func (t *Tensor) Sum(axis int) *Tensor {
	if axis < 0 || axis >= len(t.shape) {
		panic("Invalid axis")
	}

	// New shape of the resulting tensor after sum
	newShape := make([]int, len(t.shape)-1)
	copy(newShape, t.shape[:axis])
	copy(newShape[axis:], t.shape[axis+1:])

	newData := make([]float64, Product(newShape))
	indices := make([]int, len(t.shape))

	for i := 0; i < len(t.data); i++ { // Iterate through the flattened og tensor
		// Calculate the index in the result tensor
		concatIndices := append(indices[:axis], indices[axis+1:]...) // <--- This is the slice of indices for all axes except the specified summation axis
		resultIndex := Index(concatIndices, newShape)                // get flattened index of result tensor at concatIndices
		newData[resultIndex] += t.data[i]                            // <-- since we are indexing the result tensor (by excluding the summation axis), we can just add the value at the current index of the og tensor to the result tensor to sum the summmmation axis at the current index of newData
		// Increment multidimmensional indices
		for dim := len(t.shape) - 1; dim >= 0; dim-- {
			indices[dim]++
			if indices[dim] < t.shape[dim] { // If the index is within the bounds of the tensor, break
				break
			}
			indices[dim] = 0
		}
	}
	return &Tensor{shape: newShape, data: newData} // return Tensor pointer
}

// Helper function to compute the product of elements in a slice
func Product(shape []int) int {
	product := 1
	for _, dim := range shape {
		product *= dim
	}
	return product
}

// Mean() calculates the mean of elements in a tensor along a specified axis. This operation
// results in a tensor with one fewer dimension than the original tensor. It utilizes .Sum()
func (t *Tensor) Mean(axis int) *Tensor {
	sumTensor := t.Sum(axis)
	count := float64(t.shape[axis]) // <--- This is the number of elements along the specified axis
	for i := range sumTensor.data {
		sumTensor.data[i] /= count
	}
	return sumTensor
}

// Var() calculates the variance of elements in a tensor along a specified axis.
// Variance is define as the average of the squared differences from the mean
// it utilizes .Mean() within the function
func (t *Tensor) Var(axis int) *Tensor {
	if axis < 0 || axis >= len(t.shape) {
		panic("Invalid axis")
	}

	// New shape of the resulting tensor after sum
	newShape := make([]int, len(t.shape)-1)
	copy(newShape, t.shape[:axis])
	copy(newShape[axis:], t.shape[axis+1:])

	newData := make([]float64, Product(newShape))
	indices := make([]int, len(t.shape))

	meanTensor := t.Mean(axis) // compute mean tensor

	for i := 0; i < len(t.data); i++ { // Iterate through the flattened og tensor
		// Calculate the index in the result tensor
		concatIndices := append(indices[:axis], indices[axis+1:]...) // <--- This is the slice of indices for all axes except the specified summation axis
		resultIndex := Index(concatIndices, newShape)                // get flattened index of result tensor at concatIndices

		newData[resultIndex] += (t.data[i] - meanTensor.data[resultIndex]) * (t.data[i] - meanTensor.data[resultIndex]) // <-- variance formula

		// Increment multidimmensional indices
		for dim := len(t.shape) - 1; dim >= 0; dim-- {
			indices[dim]++
			if indices[dim] < t.shape[dim] { // If the index is within the bounds of the tensor, break
				break
			}
			indices[dim] = 0
		}
	}

	return &Tensor{shape: newShape, data: newData} // return Tensor pointer
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
