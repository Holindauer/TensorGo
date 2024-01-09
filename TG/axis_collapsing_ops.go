package TG

// axis_ops.go contains functions that perform operations along a specified
// axis of a tensor that result in the axis being collapsed due to the nature
// of the operation. All functions in this source file currently rely on the
// Axis_ElementOperation() function to perform this type of axis operation.

// Currently this includes: Sum_Axis(), Mean_Axis(), Var_Axis(), Std_Axis()

import (
	//"fmt"

	"math"
	"sync"
)

// ------------------------------------------------------------------------------------------------------ Axis_Collapsing_Operation

// A Tensor operation performed along an axis that results in the axis being collapsed is an Axis_Collapsing_Operation.
// For example, summing along an axis results in a tensor that is 1 dimension smaller because the axis along which the
// sum was computed is "collapsed" into that sum. This process extends to tensors of any dimmension. An intuitive way to
// think about this is to imagine that a Tensor containing N dimmensions is really a batch of Tensors that contains N-1
// dimensions. Performing an axis collapsing operation is functionally the same thinf as performing an element-wise operation
// distributed across the entire "Batch"

// The way this algorithm will work is that for a given Tensor and an axis to perform a collapsing operation along, we will first create
// a Zero_Tensor that will contain the result. Then we will iterate through that axis, create a Partial of the Tensor on that axis and
// element using Remove_Dim(). This Partial will then be passed into a go routine that will use Mutexes to add the Partial to the result
// Tensor. This will be done in parallel for each element along the axis. The result Tensor will then be returned.

// Assuming the Tensor structure and necessary helper functions like Zero_Tensor are defined elsewhere

func (A *Tensor) Axis_Collapsing_Operation(axis int, op Collapsing_Operation) *Tensor {
	if axis < 0 || axis >= len(A.Shape) {
		// Handle invalid axis error
		return nil
	}

	// Create a Zero_Tensor of the same shape as A but with the axis removed
	resultShape := make([]int, 0, len(A.Shape)-1)
	resultShape = append(resultShape, A.Shape[:axis]...)
	resultShape = append(resultShape, A.Shape[axis+1:]...)
	resultTensor := Zero_Tensor(resultShape, false)

	// Use WaitGroup and mutex to synchronize go routines
	var wg sync.WaitGroup
	var mutex sync.Mutex

	for i := 0; i < A.Shape[axis]; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Extract the partial tensor along the axis
			partial := A.Remove_Dim(axis, i)

			// Lock the mutex before modifying the result tensor
			mutex.Lock()
			op.contributeToResult(partial, resultTensor)
			mutex.Unlock()
		}(i)
	}

	wg.Wait() // Wait for all go routines to finish
	return resultTensor
}

// This interface is used to define the operation that will be performed on each partial and how it affects the result tensor.
type Collapsing_Operation interface {
	contributeToResult(partial *Tensor, result *Tensor)
}

//------------------------------------------------------------------------------------------------------ Sum_Axis()

// Collapsing_Sum_Operation defines summation on tensor elements.
type CollapsingSumOp struct{ axis int }

// contributeToResult adds each partial tensor to the result tensor.
func (s CollapsingSumOp) contributeToResult(partial, result *Tensor) {

	// indices holds the multi-dim as we iterate through the Tensor
	indices := make([]int, len(result.Shape))

	// Consider a 3x3x3 tensor. The indices will start at [0, 0, 0], [0, 0, 1], then [0, 0, 2], [0, 1, 0]... etc.
	for i := 0; i < len(result.Data); i++ {

		flatIndex := result.Index(indices) // <--- 1D index of the result tensor

		// Add the partial's element to the result tensor's element
		result.Data[flatIndex] += partial.Data[flatIndex]

		// Drecrement multi-dimensional indices.
		for dim := len(result.Shape) - 1; dim >= 0; dim-- {
			indices[dim]++
			if indices[dim] < result.Shape[dim] { // break to the next iter if we havemt reached the end of the current dimension
				break
			}
			indices[dim] = 0
		}
	}
}

// Execute is an interface used for batched operations. See batching.go for more details.
func (op CollapsingSumOp) Execute(tensors ...*Tensor) *Tensor {
	A := tensors[0]
	return A.Axis_Collapsing_Operation(op.axis, CollapsingSumOp{})
}

// Sum_Axis sums tensor elements along a specified axis.
func (A *Tensor) Sum_Axis(axis int, batching bool) *Tensor {

	// Create an instance of the CollapsingSumOp struct
	sumOp := CollapsingSumOp{axis: axis}

	if batching {
		return BatchedOperation(sumOp, A) // if batching is true, give interface to the batched execution function
	}
	return sumOp.Execute(A) // otherwise execute the interface directly

}

//------------------------------------------------------------------------------------------------------ Mean_Axis()

// Mean_Axis calculates the mean of tensor elements along a specified axis.
func (A *Tensor) Mean_Axis(axis int, batching bool) *Tensor {
	sumTensor := A.Sum_Axis(axis, batching)
	count := A.Shape[axis]
	for i := range sumTensor.Data {
		sumTensor.Data[i] /= float64(count)
	}
	return sumTensor
}

// //------------------------------------------------------------------------------------------------------ Var_Axis()

// Var_Axis calculates variance along a specified axis.
func (A *Tensor) Var_Axis(axis int, batching bool) *Tensor {

	// compute the mean and summation along the axis
	meanTensor := A.Mean_Axis(axis, false)

	// subtract the meanTensor from each element of A and square the result (elementwise)

	axes_reordering := Permute_Shape(A.Shape, axis, 0) // permute specified axis to 0 for elementwise batched summation
	A_Transposed := A.Permute(axes_reordering)

	A_diffMean := meanTensor.Broadcast_Subtract(A_Transposed)

	Squared_Differences := Multiply(A_diffMean, A_diffMean, false)

	// compute the sum of the squared differences along the 0'th axis (which was the original specified axis)
	sumSquaredDiffs := Squared_Differences.Sum_Axis(0, false)

	// divide the sum of the squared differences by the number of elements along the axis
	var inverseCount float64 = 1 / float64(A.Shape[axis])
	return sumSquaredDiffs.Scalar_Mult(inverseCount, false)
}

// //------------------------------------------------------------------------------------------------------ Std_Axis()

// StdAxisBatchOperation performs batched standard deviation calculation.
type StdAxisBatchOperation struct{ axis int }

func (op StdAxisBatchOperation) Execute(tensors ...*Tensor) *Tensor {

	A := tensors[0]

	varTensor := A.Var_Axis(op.axis, false)
	for i := range varTensor.Data {
		varTensor.Data[i] = math.Sqrt(varTensor.Data[i])
	}
	return varTensor
}

// Std_Axis calculates standard deviation along a specified axis.
func (A *Tensor) Std_Axis(axis int, batching bool) *Tensor {
	batchOp := StdAxisBatchOperation{axis: axis}
	if batching {
		return BatchedOperation(batchOp, A)
	}
	return batchOp.Execute(A)
}
