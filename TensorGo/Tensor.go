package TG

/*
* @notice tensor.go contains the Tensor struct and functions related to instantiating and retrieving data from them
* @dev The Tensor struct is the primary data structure used in the TensorGo library. It is a multi-dimensional array
* of float64 values.
* @dev Data is a 1D slice that stores multi-dimensional Tensor data contiguously.
* @dev Batched is a boolean that indicates whether the Tensor is being used as a batch of Tensors or not.
 */
type Tensor struct {
	Shape       []int
	Data        []float64
	DataReqGrad []*Value // <-- Value struct defined in AutoGrad.go
	Batched     bool
}

/*
* @notive Get() is used to access a single element from a tensor
* @param index: A slice of ints that represent the multi dimmensional index of the element to be retrieved
* @return float64: The value of the element at the given index
 */
func (A *Tensor) Get(index []int) float64 {
	// check if each index of each dim is within the bounds of the tensor
	for i, index := range index {
		if index >= A.Shape[i] {
			panic("Within Retrieve(); Index out of bounds")
		}
	}

	// Retrieve the Value struct of the element at the given index
	return A.Data[A.Index(index)]
}

/*
* @notice given a multi-dimensional index, Index() returns that elements index in the contiguously stored 1D Data slice
* @dev The algorithm for computing a flat index from a multi-dimensional index involves tje stride (number of elements to
* skip over to move one index in a given dimension) of each dimension. The stride of the last dimension is always 1.
* @param indices: A slice of ints that represent the multi dimmensional index of the element to be retrieved
* @return int: The index of the Data slice that corresponds to the given multi-dimensional index
 */
func (A *Tensor) Index(indices []int) int {

	// check that the number of indices matches the number of dimensions
	if len(indices) != len(A.Shape) {
		panic("Within Index(): Number of indices must match number of dimensions")
	}

	//create a slice of ints to store the strides
	strides := make([]int, len(A.Shape))

	// the stride of the last dimension is always 1
	strides[len(A.Shape)-1] = 1

	// decrement through Tensor axis, computing the stride of each dimension by multiplying by the stride of the next dimension
	for i := len(A.Shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * A.Shape[i+1]
	}

	// iterate through provided indices, multiplying the index by the stride of that dimension
	flatIdx := 0
	for i, index := range indices {
		flatIdx += index * strides[i]
	}

	return flatIdx
}

/*
* @notice TheoreticalIndex() is a wrapper for the Index() function that allows the computation of an index for a theoretical
* Tensor, ie based on shape.
* @param indices: A slice of ints that represent the multi dimmensional index of the element to be retrieved
* @param shape: A slice of ints that represent the dimensions of the theoretical Tensor
* @return int: The index of the Data slice that corresponds to the given multi-dimensional index
 */
func TheoreticalIndex(indices []int, shape []int) int {
	temp_tensor := &Tensor{Shape: shape} //
	return temp_tensor.Index(indices)
}

/*
* @notice UnravelIndex() converts a flat index into multi-dimensional indices based on the shape of the tensor.
* @param index: The flat index in the one-dimensional representation of the tensor.
* @return []int: The multi-dimensional indices of the element at the given flat index
 */
func (A *Tensor) UnravelIndex(index int) []int {

	// Create a slice to store the multi-dimensional indices.
	indices := make([]int, len(A.Shape))

	// Iterate over the shape in reverse order.
	for i := len(A.Shape) - 1; i >= 0; i-- {

		// axis idx for the i-th dimension is the mod of the index by the size of the i-th dimension.
		indices[i] = index % A.Shape[i]

		// Update the idx for the next iteration to be the integral counterpart of the above mod
		index = index / A.Shape[i]
	}

	// Return the calculated multi-dimensional indices.
	return indices
}

/*
* @notice getBatchElement() is used to access a single element from a batched tensor
* @param batch_element: The index of the 0'th dim of the tensor, representing the element to be retrieved
* @return *Tensor: A pointer to the Tensor that was retrieved
 */
func (A *Tensor) GetBatchElement(batch_element int) *Tensor {
	return A.Remove_Dim(0, batch_element)
}
