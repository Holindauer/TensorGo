package GLA

// This source file contians Tensor operations performed over a specified axis

import (
	"math"
)

// =========================================================================================================== Axis Operations Generalization

// Operation is an interface representing an operation on individual tensor elements
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
func (A *Tensor) AxisOperation(axis int, op Operation) *Tensor {
	if axis < 0 || axis >= len(A.Shape) { // <--- Check that the axis is valid.
		panic("Within AxisOperation(): Invalid axis")
	}

	// Calculate the shape of the result tensor.
	newShape := make([]int, len(A.Shape)-1) // <--- set new shape 1 dim smaller than original
	copy(newShape, A.Shape[:axis])          // <--- Remove specified dimension by excluding it from the copy.
	copy(newShape[axis:], A.Shape[axis+1:]) // <---

	// Initialize the data for the result tensor.
	newData := make([]float64, Product(newShape))
	indices := make([]int, len(A.Shape)) // <--- multi dim indexing for iterating through tensor

	// Perform the operation along the specified axis.
	for i := 0; i < len(A.data); i++ {
		// Concatenate the indices before and after the specified axis to form the reduced-dimension indices.
		concatIndices := append(indices[:axis], indices[axis+1:]...)

		// Convert the reduced-dimension indices to a flattened index for the result tensor.
		resultIndex := Index(concatIndices, newShape)

		// Apply the operation to the current element in the tensor and update the result tensor.
		newData[resultIndex] = op.Apply(newData[resultIndex], A.data[i])

		// Increment the multi-dimensional indices, like an odometer with each wheel representing a dimension's index.
		// For ex: in a 3x3 matrix this would be [0,0] -> [0,1] -> [0,2] -> [1,0] -> [1,1] -> [1,2] -> [2,0] -> [2,1] -> [2,2]
		for dim := len(A.Shape) - 1; dim >= 0; dim-- {
			indices[dim]++ // Advance the index in the current dimension.

			if indices[dim] < A.Shape[dim] {
				// If the index within the current dimension is still within bounds, continue with the next element.
				break
			}
			// If the current dimension's index "overflows", reset it to zero and move to increment the next higher dimension.
			indices[dim] = 0
		}
	}

	return &Tensor{Shape: newShape, data: newData}
}

// ============================================================================================================ Summation on an Axis

// This interface is used to generalize batching within any Tensor Operation. It is specific to operations that take in
// a single Tensor Argument and return a single Tensor.
type Batch_Tensor_Tensor_Interface interface {
	Execute(tensor *Tensor) *Tensor // Execute the operation and return a tensor.
}

func Batch_Tensor_Tensor_Operation(op Batch_Tensor_Tensor_Interface, A *Tensor) *Tensor {

	//define the above code as an anon func
	eachElementOp := func(example int) *Tensor {
		Batched_Output := A.Remove_Dim(example, 0)                            // <--- retrieve the first element from the 0'th dim of the batch tensor
		Output := op.Execute(Batched_Output).Add_Singleton()                  // <--- execute the operation on the first element
		singletonReordering := Indicies_First_Last_Swapped(len(Output.Shape)) // <--- swap the 0'th and len(A.Shape) - 1'th indicies
		return Output.Transpose(singletonReordering)                          // <--- reorder conitguous memory
	}

	// Start Batched Output Process by executing the operation on the first element
	Batched_Output := eachElementOp(0)

	for i := 1; i < A.Shape[0]; i++ { // <--- iterate through the remaining elements of the batch tensor
		Output := eachElementOp(i) // <--- execute the operation on the current element
		Batched_Output = Batched_Output.Concat(Output, 0)
	}

	return Batched_Output
}

// Interface for summations on a Tensor along a specified axis (using the AxisOperation() function)
type SumOperation struct{}

func (s SumOperation) Apply(a, b float64) float64 { // Apply summation on two float64 values.
	return a + b
}

// This struct is used has applied to ut the Execute() method of the Batch_Tensor_Tensor_Interface.
type SumAxisBatchOperation struct{ axis int }

// This function defines the Execute method within the Batch_Tensor_Tensor_Interface. This method is called by the
// Batch_Tensor_Tensor_Operation() function in order to performed batched Tensor summation along a specified axis.
func (op SumAxisBatchOperation) Execute(tensor *Tensor) *Tensor {
	return tensor.AxisOperation(op.axis, SumOperation{})
}

// This function Sums the elements of a Tensor along a specified axis. With the option for batched processing.
func (A *Tensor) Sum_Axis(axis int, batching bool) *Tensor {
	if !batching {
		return A.AxisOperation(axis, SumOperation{}) // Apply Axis Op to entire Tensor
	} else {
		batchOp := SumAxisBatchOperation{axis: axis}     // create a batch op struct for summation along an axis
		return Batch_Tensor_Tensor_Operation(batchOp, A) // Apply Axis Op to each element individually
	}
}

// ============================================================================================================ Mean on an Axis

// Mean calculates the mean of elements in a tensor along a specified axis
func (A *Tensor) Mean_Axis(axis int, batching bool) *Tensor {
	sumTensor := A.Sum_Axis(axis, batching) // sum along axis
	count := A.Shape[axis]
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
func (A *Tensor) Var_Axis(axis int, batching bool) *Tensor {
	meanTensor := A.Mean_Axis(axis, batching)       // mean along axis
	varOp := VarOperation{mean: meanTensor.data[0]} // pass the mean to the operation
	return A.AxisOperation(axis, varOp)             // variance along an axis
}

// ============================================================================================================ Standard Deviation on an Axis

// Std() calculates the standard deviation of elements in a tensor along a specified axis.
func (A *Tensor) Std(axis int, batching bool) *Tensor {
	varTensor := A.Var_Axis(axis, batching)
	for i := range varTensor.data {
		varTensor.data[i] = math.Sqrt(varTensor.data[i])
	}
	return varTensor
}

// ============================================================================================================ ArgMax on an Axis

// // ArgMaxOperation represents an argmax operation.
// type ArgMaxOperation struct {
// 	max float64 // used within Apply() to calculate argmax
// }

// func (a ArgMaxOperation) Apply(max, current float64) float64 { // apply argmax calculation on two float64 values.
// 	if current > a.max {
// 		return current
// 	}
// 	return max
// }

// func (A *Tensor) ArgMax_Axis(axis int) *Tensor {
// 	return A.AxisOperation(axis, ArgMaxOperation{max: math.Inf(-1)}) // argmax along an axis
// }
