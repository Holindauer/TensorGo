package main

// This source file contains functions related to manipulating the shape of a tensor.

import (
	"strconv" // <-- used to convert strings to ints
	"strings"
)

// The Partial function is used to retrieve a section out of a Tensor using Python-like slice notation.
// It accepts a Tensor and a string, then returns a pointer to a new tensor.
// Example:
// A := Range_Tensor([]int{3, 4, 9, 2})
// A_Partial := Partial(A, "0:2, 2:, :3, :")
func Partial(A *Tensor, slice string) *Tensor {
	// Remove spaces and split the slice string by commas to handle each dimension separately.
	slice = strings.ReplaceAll(slice, " ", "")
	split := strings.Split(slice, ",")
	if len(split) != len(A.shape) {
		panic("String slice arg must have the same number of dimensions as the tensor")
	}

	// Initialize slices to store the shape of the partial tensor and the start/end indices for each dimension.
	partialShape := make([]int, len(A.shape))
	partialIndices := make([][]int, len(A.shape))

	// Iterate through each dimension of the tensor to parse the slice string and compute the shape and indices of the partial tensor.
	for i, s := range split {
		start, end := 0, A.shape[i] // By default, use the entire dimension.
		if s != ":" {
			parts := strings.Split(s, ":")

			if parts[0] != "" { // If there is a start value, update start.
				start, _ = strconv.Atoi(parts[0])
			}
			if parts[1] != "" { // If there is an end value, update end.
				end, _ = strconv.Atoi(parts[1])
			}
		}
		partialShape[i] = end - start
		partialIndices[i] = []int{start, end}
	}

	// Create a new tensor to store the partial data with the computed shape.
	partialTensor := Zero_Tensor(partialShape)

	// Initialize a slice to store the current multi-dimensional index being processed.
	tempIndex := make([]int, len(partialShape))

	// Define a recursive function to fill the partial tensor.
	// The function takes the current dimension as a parameter.
	var fillPartialTensor func(int)
	fillPartialTensor = func(dim int) {
		if dim == len(partialShape) { // <--- This base case is reached for every element in the partial tensor.

			// Calculate the source index in the original tensor.
			srcIndex := make([]int, len(partialShape))
			for i, indices := range partialIndices {
				srcIndex[i] = tempIndex[i] + indices[0]
			}

			// Convert the multi-dimensional indices to flattened indices and use them to copy the data.
			srcFlattenedIndex := Index(srcIndex, A.shape)
			dstFlattenedIndex := Index(tempIndex, partialTensor.shape)
			partialTensor.data[dstFlattenedIndex] = A.data[srcFlattenedIndex]

			return
		}

		// Recursively process each index in the current dimension.
		for i := 0; i < partialShape[dim]; i++ {
			tempIndex[dim] = i
			fillPartialTensor(dim + 1)
		}
	}

	// Start the recursive process from the first dimension.
	fillPartialTensor(0)

	// Return the filled partial tensor.
	return partialTensor
}

// Reshape()  takes a tensors and a new shape for that tensors, and returns a pointer to a
// new tensors that has the same data as the original tensor, but with the new shape. Reshape
// can be done in this way becauase data for Tensors in stored contigously in memory.
func (A *Tensor) Reshape(shape []int) *Tensor {

	numElements := 1
	for _, v := range shape { // find num elements of shape param
		numElements *= v
	}
	if numElements != len(A.data) {
		panic("Cannot reshape tensor to shape with different number of elements")
	}
	// Create a new tensor to store the reshaped data with the shape param
	reshapedTensor := Zero_Tensor(shape)
	for i := 0; i < len(A.data); i++ {
		reshapedTensor.data[i] = A.data[i] // copy data from A to reshapedTensor
	}
	return reshapedTensor
}

// Transpose returns a new tensor with the axes transposed according to the given specification
// This function is modeled after the NumPy transpose function. It accepts a tensor and an array
// of integers specifying the new order of the axes. For example, if the tensor has shape [2, 3, 4]
// and the axes array is [2, 0, 1], then the resulting tensor will have shape [4, 2, 3].
func (A *Tensor) Transpose(axes []int) *Tensor {

	// Check for invalid axes
	if len(axes) != len(A.shape) {
		panic("The number of axes does not match the number of dimensions of the tensor.")
	}

	// Check for duplicate or out-of-range axes
	seen := make(map[int]bool) // map is like dict in python
	for _, axis := range axes {
		if axis < 0 || axis >= len(A.shape) || seen[axis] {
			panic("Invalid axis specification for transpose.")
		}
		seen[axis] = true
	}

	// Determine the new shape from the reordering in axes
	newShape := make([]int, len(A.shape))
	for i, axis := range axes {
		newShape[i] = A.shape[axis]
	}

	// Allocate the new tensor
	newData := make([]float64, len(A.data))
	B := &Tensor{shape: newShape, data: newData} // <-- B is a pointer to a new tensor

	// Reindex and copy data
	for i := range A.data {
		// Get the multi-dimensional indices for the current element
		originalIndices := UnravelIndex(i, A.shape)

		// Reorder the indices according to the axes array for transpose
		newIndices := make([]int, len(originalIndices))
		for j, axis := range axes {
			newIndices[j] = originalIndices[axis]
		}

		// Convert the reordered multi-dimensional indices back to a flat index
		newIndex := Index(newIndices, newShape)

		// Assign the i'th value of original tensor to the newIndex'th val of new tensor
		B.data[newIndex] = A.data[i]
	}

	return B
}
