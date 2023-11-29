package GLA

// shape.go contains functions related to manipulating the shape of a tensor.

import (
	//"fmt"
	"fmt"
	"strconv" // <-- used to convert strings to ints
	"strings"
)

//=====================================================================================================================Partial()

// The Partial function is used to retrieve a section out of a Tensor using Python-like slice notation.
// It accepts a Tensor and a string, then returns a pointer to a new tensor.
// Example:
// A := Range_Tensor([]int{3, 4, 9, 2})
// A_Partial := Partial(A, "0:2, 2:, :3, :")
func (A *Tensor) Partial(slice string) *Tensor {
	// Remove spaces and split the slice string by commas to handle each dimension separately.
	slice = strings.ReplaceAll(slice, " ", "")
	split := strings.Split(slice, ",")
	if len(split) != len(A.Shape) {
		panic("Within Partial(): String slice arg must have the same number of dimensions as the tensor")
	}

	// Initialize slices to store the shape of the partial tensor and the start/end indices for each dimension.
	partialShape := make([]int, len(A.Shape))
	partialIndices := make([][]int, len(A.Shape))

	// Iterate through each dimension of the tensor to parse the slice string and compute the shape and indices of the partial tensor.
	for i, s := range split {
		start, end := 0, A.Shape[i] // By default, use the entire dimension.
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
	partialTensor := Zero_Tensor(partialShape, false)

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
			srcFlattenedIndex := Index(srcIndex, A.Shape)
			dstFlattenedIndex := Index(tempIndex, partialTensor.Shape)
			partialTensor.Data[dstFlattenedIndex] = A.Data[srcFlattenedIndex]

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

//=====================================================================================================================Reshape()

// Reshape()  takes a tensors and a new shape for that tensors, and returns a pointer to a
// new tensors that has the same data as the original tensor, but with the new shape. Reshape
// can be done in this way becauase data for Tensors in stored contigously in memory.
func (A *Tensor) Reshape(shape []int) *Tensor {

	if Product(shape) != len(A.Data) {
		panic("Within Reshape(): Cannot reshape tensor to shape with different number of elements")
	}
	reshapedTensor := A.Copy()
	reshapedTensor.Shape = shape
	return reshapedTensor
}

//=====================================================================================================================Transpose()

// Transpose returns a new tensor with the axes transposed according to the given specification
// This function is modeled after the NumPy transpose function. It accepts a tensor and an array
// of integers specifying the new order of the axes. For example, if the tensor has shape [2, 3, 4]
// and the axes array is [2, 0, 1], then the resulting tensor will have shape [4, 2, 3].
func (A *Tensor) Transpose(axes []int) *Tensor {

	// Check for invalid axes
	if len(axes) != len(A.Shape) {
		panic("Within Transpose(): The number of axes does not match the number of dimensions of the tensor.")
	}

	// Check for duplicate or out-of-range axes
	seen := make(map[int]bool) // map is like dict in python
	for _, axis := range axes {
		if axis < 0 || axis >= len(A.Shape) || seen[axis] {
			panic("Within Transpose(): Invalid axis specification for transpose.")
		}
		seen[axis] = true
	}

	// Determine the new shape from the reordering in axes
	newShape := make([]int, len(A.Shape))
	for i, axis := range axes {
		newShape[i] = A.Shape[axis]
	}

	// Allocate the new tensor
	newData := make([]float64, len(A.Data))
	B := &Tensor{Shape: newShape, Data: newData} // <-- B is a pointer to a new tensor

	// Reindex and copy data
	for i := range A.Data {
		// Get the multi-dimensional indices for the current element
		originalIndices := UnravelIndex(i, A.Shape)

		// Reorder the indices according to the axes array for transpose
		newIndices := make([]int, len(originalIndices))
		for j, axis := range axes {
			newIndices[j] = originalIndices[axis]
		}

		// Convert the reordered multi-dimensional indices back to a flat index
		newIndex := Index(newIndices, newShape)

		// Assign the i'th value of original tensor to the newIndex'th val of new tensor
		B.Data[newIndex] = A.Data[i]
	}

	return B
}

//=====================================================================================================================Concat()

// The idea behind this algorithm stems from an understanding of how Tensor data is stored in memory.
// Tensors of n dimmension are stored contiguously in memory as a 1D array. The multi-dimensionality
// of the tensor is simulated by indexing the 1D array using a strided index. This means that if you
// are atttemping to index a 5D tensor of shape [3, 3, 3, 3, 3], and you want to move one element up
// the last dimmension, then you must 'stride' over all elements of the 4th dimmension stored in the
// contigous memory to get there. This task is handled by the Index() and Retrieve() functions.
// ---------------------------------------------------------------------------------------------------
// This way of storing data in in memory introduces complexity when concatenating tenosrs along an axis.
// When the axis of concatenation is the 0'th axis, the algorithm is simple. No striding is required, and
// the contigous data from one tensor can just be appended to the other.
// ---------------------------------------------------------------------------------------------------
// However, when the axis of concatenation is not the 0'th axis, the algorithm becomes more complex
// due to the striding. This algorithm handles this complexity by simplifying the cases where the axis
// of concatenation is not zero by first tranpsosing the tensors such that the axis of concatenation
// is the 0'th axis. They can then simply be appended together contiguously and transposed back to the
// original ordering of dimmensions.
func (A *Tensor) Concat(B *Tensor, axis_cat int) *Tensor {

	Check_Concat_Requirements(A, B, axis_cat)

	var concatTensor *Tensor
	if axis_cat == 0 { // handle axis of concatenation at 0'th axis

		// Determine the shape of the concatenated tensor
		concatShape := make([]int, len(A.Shape))
		for i := 0; i < len(A.Shape); i++ {
			if i == axis_cat {
				concatShape[i] = A.Shape[i] + B.Shape[i] // <--- concatenation extends this dimension
			} else {
				concatShape[i] = A.Shape[i]
			}
		}

		// concatenate data contiguously into new slice
		concatData := append(A.Data, B.Data...)

		// create new tensor to store concatenated data for return
		concatTensor = &Tensor{Shape: concatShape, Data: concatData}
	} else if axis_cat != 0 {

		// determine the reordering of the axes for transpose to make axis_cat the 0'th axis the slice
		// will be a permutation of the numbers 0 through len(A.shape) - 1 with axis cat and 0 swapped
		axes_reordering := Permute_Shape(A.Shape, axis_cat, 0)

		// transpose A and B to make axis_cat the 0'th axis
		A_T := A.Transpose(axes_reordering)
		B_T := B.Transpose(axes_reordering)

		// concatenate data contiguously into new slice
		concatData_Transposed := append(A_T.Data, B_T.Data...)

		// We now have a slice of contigous data that is the concatenation of A_T and B_T, in order to use
		// this data to create a new tensor, we must first determine the shape of the new tensor in this
		// Trasnposed form. This can be done by copying A_T.shape and adding B_T.shape[0] to it.
		concatShape_Transposed := make([]int, len(A_T.Shape))
		for i := 0; i < len(A_T.Shape); i++ {
			if i == 0 {
				concatShape_Transposed[i] = A_T.Shape[i] + B_T.Shape[i]
			} else {
				concatShape_Transposed[i] = A_T.Shape[i]
			}
		}

		// create new tensor to store the transposed concatenated data
		concatTensor_Transposed := &Tensor{Shape: concatShape_Transposed, Data: concatData_Transposed}

		// transpose the concatenated tensor back to the original ordering of axes. Because we only swapped
		// two axes, we can just reuse the same axe_reordering array from the originbal transpose.
		concatTensor = concatTensor_Transposed.Transpose(axes_reordering)
	}

	return concatTensor
}

// Permute_Shape creates a new order for axes to transpose a tensor by swapping two specified axes.
// This function generates a permutation of the axis indices of a tensor's shape, where the two specified axes are swapped,
// and the rest of the axes retain their original order.
//
// Parameters:
// - shape: A slice of integers representing the shape of the tensor.
// - axis1, axis2: The two axes to be swapped.
//
// Returns:
// A slice of integers representing the reordered axes for transposition.
func Permute_Shape(shape []int, axis1, axis2 int) []int {
	if axis1 < 0 || axis1 >= len(shape) || axis2 < 0 || axis2 >= len(shape) {
		panic("SwapAxesForTranspose --- Invalid axes")
	}

	axesReordering := make([]int, len(shape))
	for i := range shape {
		axesReordering[i] = i
	}

	// Swap the two axes
	axesReordering[axis1], axesReordering[axis2] = axesReordering[axis2], axesReordering[axis1]

	return axesReordering
}

func Check_Concat_Requirements(A *Tensor, B *Tensor, axis_cat int) {
	// Ensure that the number of dimensions of the tensors are the same
	if len(A.Shape) != len(B.Shape) {
		panic("Within Concat(): The number of dimensions of the tensors must be the same.")
	}

	// Check that axis_cat is within the valid range
	if axis_cat < 0 || axis_cat >= len(A.Shape) {
		panic("Within Concat(): axis_cat is out of bounds for the shape of the tensors.")
	}

	// Ensure that the shape of the tensors are the same except for the axis of concatenation
	for i := 0; i < len(A.Shape); i++ {
		if i != axis_cat && A.Shape[i] != B.Shape[i] {
			panic("Within Concat(): The shapes of the tensors must be the same except for the axis of concatenation.")
		}
	}
}

//=====================================================================================================================Extend_Shape()

// The Extend() method is used to add a new dimmension to the tensor. The new dimmension each element
// across the new dimmension contains a state of the pre extended tensor with all other dimmension elements
// copied into it. The new dimmension is added to the end of the shape of the tensor. The Extend() method
// returns a pointer to a new tensor with the extended shape and zeroed data.
func (A *Tensor) Extend_Shape(num_elements int) *Tensor {
	// Check that the number of elements is valid
	if num_elements < 1 {
		panic("Within Extend_Shape(): The number of elements must be positive.")
	}
	newShape := make([]int, len(A.Shape)+1) // add dim
	copy(newShape, A.Shape)
	newShape[len(A.Shape)] = num_elements // set new dim num_elements

	// Create a new tensor with the extended shape and zeroed data
	extendedTensor := Zero_Tensor(newShape, false)

	// Fill the extended tensor with data from the original tensor
	// Initialize a slice to store the current multi-dimensional index for the new tensor
	tempIndex := make([]int, len(newShape))

	// Recursive function to fill the extended tensor
	var fillExtendedTensor func(int)
	fillExtendedTensor = func(dim int) {
		if dim == len(A.Shape) { // If we reached the last original dimension
			// Copy the data from the original tensor to the extended tensor
			srcFlattenedIndex := Index(tempIndex[:len(tempIndex)-1], A.Shape)
			for i := 0; i < num_elements; i++ {
				tempIndex[len(tempIndex)-1] = i
				dstFlattenedIndex := Index(tempIndex, newShape)
				extendedTensor.Data[dstFlattenedIndex] = A.Data[srcFlattenedIndex]
			}
			return
		}

		// Recursively process each index in the current dimension
		for i := 0; i < A.Shape[dim]; i++ {
			tempIndex[dim] = i
			fillExtendedTensor(dim + 1)
		}
	}

	// Start the recursive process from the first dimension
	fillExtendedTensor(0)

	// Return the filled extended tensor
	return extendedTensor
}

//=====================================================================================================================Extend_Dim()

// The Extend_Dim() method is used to add new elements to an already existing axis within a tensor.
// The new data will be initialized to zero. The integer argument axis specifies the axis to be extended,
// and the integer argument num_elements specifies the number of zeroed elements to be added to the axis.
// The Extend_Dim() method returns a pointer to a new tensor with the extended shape and zeroed data.
func (A *Tensor) Extend_Dim(axis int, num_elements int) *Tensor {

	Check_Extend_Dim_Requirements(A, axis, num_elements)

	// Create a new shape with extended dimmension
	newShape := make([]int, len(A.Shape))
	copy(newShape, A.Shape)                       // <--- Copy the original shape
	newShape[axis] = num_elements + A.Shape[axis] // <--- Add the new dimension to axis

	// Create a new tensor with the extended shape and zeroed data
	extendedTensor := Zero_Tensor(newShape, false)

	// Next is to fill the extended tensor with data from the original tensor. First,
	// Initialize a slice to store the current multi-dimensional index for the new tensor
	tempIndex := make([]int, len(newShape))

	// Recursive function to fill the extended tensor
	var fillExtendedTensor func(int)
	fillExtendedTensor = func(dim int) {
		if dim >= len(A.Shape) {
			// As the recursion unwinds, this base case is reached where we copy data from the original tensor in the appropriate idx
			srcFlattenedIndex := Index(tempIndex, A.Shape)  // <---  Index() call for og vs dest differ by shape provided as arg
			dstFlattenedIndex := Index(tempIndex, newShape) // <---
			extendedTensor.Data[dstFlattenedIndex] = A.Data[srcFlattenedIndex]
			return
		}

		// Recursively process each index in the current dimension  By default the new tensors have
		// zeroed data, so we only need to copy data from the original tensor @ appropriate indices
		// each recursive call iterates over all elements within a single dimmension of the tensor
		for i := 0; i < A.Shape[dim]; i++ {
			tempIndex[dim] = i
			fillExtendedTensor(dim + 1)
		}
	}

	// Start the recursive process from the first dimension
	fillExtendedTensor(0)

	return extendedTensor
}

func Check_Extend_Dim_Requirements(A *Tensor, axis int, num_elements int) {
	if axis < 0 || axis >= len(A.Shape) {
		panic("Within Extend_Dim(): The axis is out of bounds for the shape of the tensor.")
	}
	if num_elements < 1 {
		panic("Within Extend_Dim(): The number of elements must be positive and greater than 0.")
	}
}

//=====================================================================================================================Remove_Dim()

// This function uses the Partial() method to take remove remove an axis from a tensor by taking the Partial
// with all dims kept the same, except for one becoming a singleton dimmension. This requires  an argument of
// the axis in which to remove the dim as well as which index of that axis to keep. Remove_Dim() first checks.
func (A *Tensor) Remove_Dim(axis_of_removal int, element_of_retrieval int) *Tensor {
	// create an empty string to store the slice string for Partial()
	var builder strings.Builder

	// Iterate through the shape of the tensor, appending ":," if the dim is not the element of retrieval
	// and element_of_retrieval : (element_of_retrieval + 1) if it is the element of retrieval
	for i := range A.Shape {
		if i == axis_of_removal {
			// Add the specific index for the axis of removal
			builder.WriteString(fmt.Sprintf("%d:%d,", element_of_retrieval, element_of_retrieval+1))
		} else {
			// Add ':' to keep the entire dimension
			builder.WriteString(":,")
		}
	}

	// Remove the trailing comma
	sliceString := strings.TrimRight(builder.String(), ",")

	// Use the Partial() method to take a partial tensor
	A_Partial := A.Partial(sliceString).Remove_Singletons()

	return A_Partial
}

//=====================================================================================================================Remove_Singleton()

// This function is used to remove a singleton dimmension from a Tensor, It will remove all singleton dimmensions
// it finds. Which essentially just means that it adjusts the shape slice of the tensor to remove elements of val 1
func (A *Tensor) Remove_Singletons() *Tensor {

	// initialize a slice to store the new shape of the tensor
	squeezedShape := make([]int, 0)

	// iterate through the shape of the tensor and append all elements that are not 1 to newShape
	for _, dim := range A.Shape {
		if dim != 1 {
			squeezedShape = append(squeezedShape, dim)
		}
	}

	// create a new tensor with the new shape and copy the data from A to the new tensor
	newTensor := A.Copy()
	newTensor.Shape = squeezedShape // <--- Replace the old shape with the squeezed shape

	return newTensor

}

//=====================================================================================================================Add_Singleton()

// This function is used to add a singleton dimmension to a Tensor, this menas that a 1 is simply appended to the end
// of the shape of the existing Tensor. A pointer to a new Tensor is return.
func (A *Tensor) Add_Singleton() *Tensor {

	newTensor := A.Copy()
	newTensor.Shape = append(newTensor.Shape, 1) // <--- Append a 1 to the end of the shape slice

	return newTensor
}
