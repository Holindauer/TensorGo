package TG

/* @notice shape_ops.go contain functions that manipulate a tensors shape in some way.
* @def Shape ops manipulate the shape exclusively, they do not change the underlying data in any way.
* with the exception of reording contiguous data to reflect changes in multi dimensional shape
 */

import (
	"fmt"
	"strconv"
	"strings"
)

//============================================================================================================================== Slice()

/*
* @notice Slice() uses Python-like slice notation to retrieve a subset of a Tensor. A slice from each dimension of the tensor
* is specified by a colon-separated string. For example, the string "1:3, 2:4" would retrieve a 2x2 slice from a 5x5 tensor.
* @dev slice notation is exclusive of the index to the right of the colon.
* @dev if an end index is not specified, the slice will extend to the end of the dimension.
* @dev RequiresGrad and Batched flags are preserved in the sliced tensor.
* @param slice: A string containing the slice notation for each dimension of the tensor.
* @return *Tensor: A pointer to a new tensor containing the sliced data.
 */
func (A *Tensor) Slice(slice string) *Tensor {

	// Remove spaces and split the slice string by commas to handle each dimension separately.
	slice = strings.ReplaceAll(slice, " ", "")
	split := strings.Split(slice, ",")
	if len(split) != len(A.Shape) {
		panic("Within Partial(): String slice arg must have the same number of dimensions as the tensor")
	}

	// Initialize slices to store the shape of the partial tensor and the start/end indices for each dimension.
	sliceShape := make([]int, len(A.Shape))
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
		sliceShape[i] = end - start
		partialIndices[i] = []int{start, end}
	}

	// Create a new tensor to store the partial data with the computed shape.
	slicedTensor := ZeroTensor(sliceShape, false)
	slicedTensor.Batched = A.Batched

	slicedTensor.RequireGrad = true

	// Initialize a slice to store the current multi-dimensional index being processed.
	tempIndex := make([]int, len(sliceShape))

	// Define a recursive anon function to fill the partial tensor.
	// The function accepts the current dimension as a parameter.
	var fillPartialTensor func(int)
	fillPartialTensor = func(dim int) {
		if dim == len(sliceShape) { // <--- This base case is reached for every element in the slice tensor.

			// Calculate the source index in the original tensor.
			srcIndex := make([]int, len(sliceShape))
			for i, indices := range partialIndices {
				srcIndex[i] = tempIndex[i] + indices[0]
			}

			// Convert the multi-dimensional indices to flattened indices and use them to copy the data.
			srcFlattenedIndex := A.Index(srcIndex)
			dstFlattenedIndex := slicedTensor.Index(tempIndex)

			if A.RequireGrad {
				slicedTensor.DataReqGrad[dstFlattenedIndex].Scalar = A.DataReqGrad[srcFlattenedIndex].Scalar
			}
			slicedTensor.Data[dstFlattenedIndex] = A.Data[srcFlattenedIndex]

			return
		}

		// Recursively process each index in the current dimension.
		for i := 0; i < sliceShape[dim]; i++ {
			tempIndex[dim] = i
			fillPartialTensor(dim + 1)
		}
	}

	// Start the recursive process from the first dimension.
	fillPartialTensor(0)

	return slicedTensor
}

//============================================================================================================================== Reshape()

// / @dev This struct contains the new shape me
type ReshapeOp struct{ shape []int }

/*
* @notice The execute method of the ReshapeOp struct is used to apply the reshape operation to a single Tensor.
* @dev ReshapeOps Execute method can also be sent to the BatchedOperation func in order to reshape individual
* batch elements while maintaining the batch dim.
 */
func (op ReshapeOp) Execute(tensors ...*Tensor) *Tensor {

	// extract tensor arg
	A := tensors[0]

	// Ensure the Tensor is being reshaped to a valid dimmension
	if Product(op.shape) != len(A.Data) {
		panic("Within Reshape(): Cannot reshape tensor to shape with different number of elements")
	}

	// Set the new shape in A
	A.Shape = op.shape

	// Return the pointer to A
	return A
}

/*
* @notice Reshape changes the shape of a Tensor to a different shape, this does not include manipulating
* the underlying continuous memory.
* @dev the product of all terms in the provided integer slice argument must math the length of the
* data in contiguous memory
* @param A is a Tensor pointer to change the shape of.
* @param shape is an integer slice representing the new shape fo the Tensor
 */
func (A *Tensor) Reshape(shape []int, batching bool) *Tensor {

	// Setup the ReshapeOp struct with the shape arg
	reshape := ReshapeOp{shape: shape}

	if batching {
		return BatchedOperation(reshape, A) // batched reshape
	}
	return reshape.Execute(A) // otherwise single reshape
}

//============================================================================================================================== Transpose()

/*
* @notice Permute is used to reorder the dimmension of a Tensor.
* @dev this reordering happens on both the level of the integer shape slice as well as on the level
* of the contiguous memory
* @param A is a pointer to the Tensor to permute
* @param permutaion is an integer slice with elements from 0 to [len(A.shape) - 1] in any order.
* For example: [0, 3, 2, 1, 4] will reorder the dimmensions [3, 4, 5, 7, 5] to [3, 7, 5, 4, 5]
 */
func (A *Tensor) Permute(perumuation []int) *Tensor {

	// Check for invalid axes
	if len(perumuation) != len(A.Shape) {
		panic("Within Transpose(): The number of axes does not match the number of dimensions of the tensor.")
	}

	// Check for duplicate or out-of-range axes
	seen := make(map[int]bool) // map is like dict in python
	for _, axis := range perumuation {
		if axis < 0 || axis >= len(A.Shape) || seen[axis] {
			panic("Within Transpose(): Invalid axis specification for transpose.")
		}
		seen[axis] = true
	}

	// Determine the new shape from the reordering in axes
	newShape := make([]int, len(A.Shape))
	for i, axis := range perumuation {
		newShape[i] = A.Shape[axis]
	}

	// Create a Zero Tensor
	B := ZeroTensor(newShape, false)

	// Reindex and copy data
	for i := range A.Data {
		// Get the multi-dimensional indices for the current element
		originalIndices := A.UnravelIndex(i)

		// Reorder the indices according to the axes array for transpose
		newIndices := make([]int, len(originalIndices))
		for j, axis := range perumuation {
			newIndices[j] = originalIndices[axis]
		}

		// Convert the reordered multi-dimensional indices back to a flat index
		newIndex := TheoreticalIndex(newIndices, newShape)

		// Assign the i'th value of original tensor to the newIndex'th val of new tensor
		B.Data[newIndex] = A.Data[i]
	}

	return B
}

//============================================================================================================================== Concat()

/*
* @notice Concat is used to Concatenate a Tensor to another Tensor along a particular axis of concatenation
* @dev In order to concatenate a Tensor to another Tensor along a partical axes, all axes except the axis
* of concatenation must be the same.
* @dev when the axis of concatenation is 0, the contiguous data can just be appended from one Tensor to the
* other and have the shape adjust
* @dev when the axis fo concatenation is not 0, the Tensor is permuted such that the axis of concatenation is
* the 0'th axis. Then he contiguous memory is appened, shape adjusted, and the Tensor is then permuted back to
* the original configuration.
 */
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
		A_T := A.Permute(axes_reordering)
		B_T := B.Permute(axes_reordering)

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
		concatTensor = concatTensor_Transposed.Permute(axes_reordering)
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

/*
* @notice Permute Shape is a helper function that accepts an integer slice representing a Tensor shape and and two
* indices of the shape slice that are to be swapped.
*
 */
func Permute_Shape(shape []int, axis1, axis2 int) []int {

	// Check that the axes are valid for the number of dims in the shape
	if axis1 < 0 || axis1 >= len(shape) || axis2 < 0 || axis2 >= len(shape) {
		panic("Permute_Shape -- Invalid axes provided")
	}

	// Initialize a slice to store the new axes order
	axesReordering := make([]int, len(shape))
	for i := range shape {
		axesReordering[i] = i
	}

	// Swap the two axes
	axesReordering[axis1], axesReordering[axis2] = axesReordering[axis2], axesReordering[axis1]

	return axesReordering
}

/*
* @notice Check_Concat_Requirements is a helper function that checks that the requirements for concatenation
* are met. This includes that the number of dimensions of the tensors are the same, that the axis of
* concatenation is within the valid range, and that the shape of the tensors are the same except for the axis
* of concatenation.
 */
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

//============================================================================================================================== Extend_Shape()

/*
* @notice Extend() method is used to add a new dimmension to the tensor.
* @dev Each element of the extended Tensor contains  the state of the original Tensor with all other dimmension
* elements copied into it. The new dimmension is added to the end of the shape of the tensor.
* @param num_elements is the number of elements to extend the new dim by
* @returns pointer to the new Tensor
 */
func (A *Tensor) Extend_Shape(num_elements int) *Tensor {

	// Check that the number of elements is valid
	if num_elements < 1 {
		panic("Within Extend_Shape(): The number of elements must be positive.")
	}
	newShape := make([]int, len(A.Shape)+1) // add dim
	copy(newShape, A.Shape)
	newShape[len(A.Shape)] = num_elements // set new dim num_elements

	// Create a new tensor with the extended shape and zeroed data
	extendedTensor := ZeroTensor(newShape, false)

	// Fill the extended tensor with data from the original tensor
	// Initialize a slice to store the current multi-dimensional index for the new tensor
	tempIndex := make([]int, len(newShape))

	// Recursive function to fill the extended tensor
	var fillExtendedTensor func(int)
	fillExtendedTensor = func(dim int) {
		if dim == len(A.Shape) { // If we reached the last original dimension
			// Copy the data from the original tensor to the extended tensor
			srcFlattenedIndex := A.Index(tempIndex[:len(tempIndex)-1])
			for i := 0; i < num_elements; i++ {
				tempIndex[len(tempIndex)-1] = i
				dstFlattenedIndex := TheoreticalIndex(tempIndex, newShape)

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

//============================================================================================================================== Extend_Dim()

/*
* @notice Extend_Dim() method is used to add new elements to an already existing axis within a tensor.
* @dev The new data will be initialized to zero.
* @param axis specifies the axis to be extended,
* @param num_elements specifies the number of zeroed elements to be added to the axis.
* @returns a pointer to a new tensor with the extended shape and zeroed data.
 */
func (A *Tensor) Extend_Dim(axis int, num_elements int) *Tensor {

	Check_Extend_Dim_Requirements(A, axis, num_elements)

	// Create a new shape with extended dimmension
	newShape := make([]int, len(A.Shape))
	copy(newShape, A.Shape)                       // <--- Copy the original shape
	newShape[axis] = num_elements + A.Shape[axis] // <--- Add the new dimension to axis

	// Create a new tensor with the extended shape and zeroed data
	extendedTensor := ZeroTensor(newShape, false)

	// Next is to fill the extended tensor with data from the original tensor. First,
	// Initialize a slice to store the current multi-dimensional index for the new tensor
	tempIndex := make([]int, len(newShape))

	// Recursive function to fill the extended tensor
	var fillExtendedTensor func(int)
	fillExtendedTensor = func(dim int) {
		if dim >= len(A.Shape) {
			// As the recursion unwinds, this base case is reached where we copy data from the original tensor in the appropriate idx
			srcFlattenedIndex := TheoreticalIndex(tempIndex, A.Shape)  // <---  Index() call for og vs dest differ by shape provided as arg
			dstFlattenedIndex := TheoreticalIndex(tempIndex, newShape) // <---
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

//============================================================================================================================== Remove_Dim()

/*
* @notice Remove dim removes an axis from a Tensor
* @dev In order to remove an entire dimmension, we must specify which element of the axis we are removing to keep.
* This is done usiong the Slice() function.
* @param axis_of_removal is the axis to remove
* @param element_of_retrieval is the element of the axis to keep
* @returns a pointer to a new tensor with the specified axis removed
 */
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

	// Use the Slice() method to take a partial tensor
	A_Partial := A.Slice(sliceString).Remove_Singletons()

	return A_Partial
}

//============================================================================================================================== Remove_Singleton()

/*
* @notice Remove_Singleton() removes all singleton dimmensions from a Tensor
 */
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

//============================================================================================================================== Add_Singleton()

/*
* @notice Add_Singleton() adds a singleton dimmension to a Tensor
* @param index is the index of the shape slice to add the singleton to
* @returns a pointer to a new tensor with the singleton dimmension added
 */
func (A *Tensor) Add_Singleton(index int) *Tensor {

	newTensor := A.Copy()
	newTensor.Shape = append(newTensor.Shape, 1) // <--- Append a 1 to the end of the shape slice

	// insert a 1 to specified index
	if index != 0 {
		copy(newTensor.Shape[index+1:], newTensor.Shape[index:])
		newTensor.Shape[index] = 1
	}

	return newTensor
}
