package main

// This source file contains functions related to manipulating the shape of a tensor.

import (
	"fmt"
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

// tenors_1.Concat(tensor_2, axi_of_concatenation) returns a new tensor that is the concatenation
// of the two tensors along the specified axis. The axis of concatenation must have the same length
// for both tensors, but all other axes must be the same length.
func (A *Tensor) Concat(B *Tensor, axis_cat int) *Tensor {

	// Ensure that the number of dimensions of the tensors are the same
	if len(A.shape) != len(B.shape) {
		panic("The number of dimensions of the tensors must be the same.")
	}

	// Ensure that the shape of the tensors are the same except for the axis of concatenation
	for i := 0; i < len(A.shape); i++ {
		if i != axis_cat && A.shape[i] != B.shape[i] { // Condition is satisfied when i'th dim  of A and B are not equal and i != axis_cat ,, ie the shapes are not the same except for the axis of concatenation
			panic("The shapes of the tensors must be the same except for the axis of concatenation.")
		}
	}

	// Determine the shape of the concatenated tensor
	concatShape := make([]int, len(A.shape))
	for i := 0; i < len(A.shape); i++ {
		if i == axis_cat {
			concatShape[i] = A.shape[i] + B.shape[i] // <--- concatenation extends this dimension
		} else {
			concatShape[i] = A.shape[i]
		}
	}

	// This algorithm relies on Transposing Tensor A and B such that the axis of concatenation is the last axis
	// for them both. We'll use the Transpose() method to do this, which takes an array of integers specifying
	// the reordering of the axes. For example, if the tensor has shape [2, 3, 4] and the axes array is [2, 0, 1],

	// Here we are reordering the axes such that the axis of concatenation is the last axis
	axes_reordering := make([]int, len(A.shape))
	for i := 0; i < len(A.shape); i++ {
		axes_reordering[i] = i
	}

	// create a copy of the original axis before reordering
	original_axis_order := make([]int, len(A.shape))
	copy(original_axis_order, axes_reordering)

	// If the axis of concatenation is the last axis of the tensor, then we don't need to do anything.
	if axis_cat != len(A.shape)-1 {
		axes_reordering[axis_cat] = 0
		axes_reordering[0] = axis_cat
	} else if axis_cat == len(A.shape)-1 {
		axes_reordering[axis_cat] = len(A.shape) - 1
		axes_reordering[len(A.shape)-1] = axis_cat

	}

	// Print A and B before Transposing
	fmt.Println("Before Transposing")
	Display_Matrix(A)
	fmt.Println("")
	Display_Matrix(B)
	fmt.Println("")

	// Transpose A and B with the axes reordering
	A_T := A.Transpose(axes_reordering)
	B_T := B.Transpose(axes_reordering)

	// Print A_T and B_T after Transposing
	fmt.Println("After Transposing")
	Display_Matrix(A_T)
	fmt.Println("")
	Display_Matrix(B_T)
	fmt.Println("")

	// Because we've moved the Axis of concatenation to the last axis, appening the contigous
	// data does not require any striding of the multidimensional data. We can simply append
	// the data from A_T and B_T to create the Tranposed concatenated data.
	concatData := append(A_T.data, B_T.data...) // <--- the dots unpack the elements of B_T.data

	// create new tensor pointer with the concatenated data and shape
	concatTensor := &Tensor{shape: concatShape, data: concatData}

	// Print the concatenated tensor
	fmt.Println("Concatenated Tensor")
	Display_Matrix(concatTensor)
	fmt.Println("")

	// Return the Transpose of the concatenated tensor with the original axis order
	return concatTensor.Transpose(original_axis_order)
}
