package main

import (
	"strconv" // <-- used to convert strings to ints
	"strings"
)

// This file contains functions for Tensor Operations, irregardless of dimensionality

// This funciton performs scalar multiplication on a tensor in place
// It returns a pointer to the same tensor
func Scalar_Mult_(A *Tensor, scalar float64) *Tensor {

	for i := 0; i < len(A.data); i++ {
		A.data[i] *= scalar
	}

	return A // <-- Pointer to the same tensor
}

// This function performs elementwise addition on two tensors
// The tensors must have the same shape. It returns a pointer to a new tensor
func Add(A *Tensor, B *Tensor) *Tensor {

	// Check that the tensors have the same shape
	if !Same_Shape(A, B) {
		panic("Tensors must have the same shape")
	}

	// Create a new tensor to hold the result
	C := Zero_Tensor(A.shape)

	// Perform the elementwise addition
	for i := 0; i < len(A.data); i++ {
		C.data[i] = A.data[i] + B.data[i]
	}

	return C // <-- Pointer to the new tensor
}

// This function performs elementwise subtraction on two tensors
// The tensors must have the same shape. It returns a pointer to a new tensor
func Subtract(A *Tensor, B *Tensor) *Tensor {

	// Check that the tensors have the same shape
	if !Same_Shape(A, B) {
		panic("Tensors must have the same shape")
	}

	// Create a new tensor to hold the result
	C := Zero_Tensor(A.shape)

	// Perform the elementwise subtraction
	for i := 0; i < len(A.data); i++ {
		C.data[i] = A.data[i] - B.data[i]
	}

	return C // <-- Pointer to the new tensor
}

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

// To implement:

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

// Functions I am planning to implement are listed below:

// einsum

// transpose --- recievs (2, 3, 1, 0) ie a new ordering of dims
// manipualtes underlying contiugous data to return a new tensor with the new ordering

// various statistical functions
// mean std var sum prod

// unique elements of array

// argmax along a dimension

// argmin along a dimension

// covariance matrix computation

// normalization functions --- implement a few major strategies

// concatenate along an axis
