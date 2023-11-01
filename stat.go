package main

// this source file contatins statistical functions for tensors

// import (
// 	"sync"
// )

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
	return &Tensor{shape: newShape, data: newData}
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
