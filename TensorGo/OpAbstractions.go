package TG

import (
	"sync"
)

/*
* @notice OpAbstractions.go contains functions that abstract the execution of specific types of oepraitons on tensors
* This frees up the devolopment of specific types  (closed/open, unary/binary/... )of tensor from the interfaces that make them possible.
*
 */

//============================================================================================================================== Elementwise Tensor Operations

// This interace is used to generalize elementwise tensor operations on the level of individual elements
type _ElementwiseOp interface {
	ExecuteElementwiseOp(a, b float64) float64
}

// This function is a generalization of elementwise tensor operations. It takes in two tensors and an Element_Operation.
func ElementwiseOp(A *Tensor, B *Tensor, op _ElementwiseOp) *Tensor {

	if !Same_Shape(A, B) {
		panic("Within Elementwise_Operation(): Tensors must have the same shape")
	}

	C := ZeroTensor(A.Shape, false)
	for i := 0; i < len(A.Data); i++ {
		C.Data[i] = op.ExecuteElementwiseOp(A.Data[i], B.Data[i]) // perform operation with elements
	}
	return C
}

//============================================================================================================================== Gradient Tracked Elementwise Tensor Operations

// This interace is used to generalize elementwise tensor operations on the level of individual elements
type _ElementwiseOpGrad interface {
	ExecuteElementwiseOp(a, b *Value) *Value
}

// This function is a generalization of elementwise tensor operations. It takes in two tensors and an Element_Operation.
func ElementwiseOpGrad(A *Tensor, B *Tensor, op _ElementwiseOpGrad) *Tensor {

	if !Same_Shape(A, B) {
		panic("Within Elementwise_Operation(): Tensors must have the same shape")
	}

	C := ZeroTensor(A.Shape, false)

	for i := 0; i < len(A.Data); i++ {
		C.DataReqGrad[i] = op.ExecuteElementwiseOp(A.DataReqGrad[i], B.DataReqGrad[i]) // perform operation with elements
	}
	return C
}

//============================================================================================================================== Broadcasting

/*
* @notice broadcasting is when an operation that requires two tensors of the same shape is performed with two tensors,
* where one is of a dimmenionality one less than the other. The tensor of lower dimensionality is "broadcasted to each
* the elements in the dimmension it lacks. Broadcast() performs this operation generally.
* @param BroadcastArg is the Tensor of dimensionality one less than Broadcast_Onto
* @param BroadcastOnto is the Tensor of dimensionality one greater than BroadcastArg
* @param op is the function that will is desired to performed in a broadcasted manner
 */
func (BroadcastArg *Tensor) Broadcast(BroadcastOnto *Tensor, op func(onto *Tensor, arg *Tensor) *Tensor) *Tensor {

	if !isEqual(BroadcastArg.Shape, BroadcastOnto.Shape[1:]) {
		panic("Broadcast() requires that the shape of the first Tensor be equal to the shape of the second Tensor with the first axis removed")
	}

	// Add a singleton to the beginning of the shape of Broadcast_Arg so that it can be concatenated upon
	Broadcast_Arg_Copy := BroadcastArg.Copy() // we are copying the Tensor so we don't want to modify the original
	Broadcast_Arg_Copy.Shape = append([]int{1}, BroadcastArg.Shape...)

	// For the first element along the 0'th axis of Broadcast_Onto, apply the operation
	result := BroadcastOnto.Remove_Dim(0, 0).Reshape(Broadcast_Arg_Copy.Shape, false)
	result = op(result, Broadcast_Arg_Copy) // apply the operation to the first element along the 0'th axis

	for i := 1; i < BroadcastOnto.Shape[0]; i++ {

		elementResult := BroadcastOnto.Remove_Dim(0, i).Reshape(Broadcast_Arg_Copy.Shape, false)

		result = result.Concat(op(elementResult, Broadcast_Arg_Copy), 0)
	}
	return result
}

//============================================================================================================================== Operations That Collapse a Tensor into a Scalar

// AllOperation is an interface representing an operation applied to all elements of a tensor.
type ScalarCollapsingOp interface {
	Apply(*Tensor, int, int) float64  // Apply performs the operation on a chunk of the tensor's data.
	CombineResults([]float64) float64 // CombineResults combines the results from all chunks.
}

/*
* @notice ScalarCollapseOp is an Operation generalization for functions that collapse a Tensor into a Scalar
* @dev uses concurrency to speed up the process, Apply() is used to apply the provided operation to a chunk of the Tensor's data
* @dev CombineResults() is used to combine the results from all chunks from goroutines
* @param op is the operation to be applied to all elements of the Tensor
* @returns a float64 that is the result of applying the operation
 */
func (t *Tensor) ScalarCollapseOp(op ScalarCollapsingOp) float64 {
	var wg sync.WaitGroup
	var mutex = &sync.Mutex{}

	numGoroutines := 4
	chunkSize := len(t.Data) / numGoroutines
	results := make([]float64, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)

		start := i * chunkSize
		end := start + chunkSize

		if i == numGoroutines-1 {
			end = len(t.Data) // Ensure the last chunk includes any remaining elements
		}

		go func(i int, start, end int) {
			defer wg.Done()

			chunkResult := op.Apply(t, start, end) // Apply the operation to the chunk of data.

			mutex.Lock()
			results[i] = chunkResult
			mutex.Unlock()
		}(i, start, end)
	}
	wg.Wait()

	return op.CombineResults(results) // Combine the results from all chunks.
}

//============================================================================================================================== AxisInplaceOperation()

/*
* @notice AxisElementwiseOperation() applies an operation to each element along a given axis of a Tensor. This results in a Tensor that
 */
type InplaceOperation interface {
	Apply_InplaceOp(*Tensor)
}

// BatchInplaceOperation applies an Inplace_Operation to a slice of the tensor
type BatchInplaceOperation struct {
	op InplaceOperation
}

func (b *BatchInplaceOperation) Execute(tensors ...*Tensor) *Tensor {

	A := tensors[0]

	// Apply the Inplace_Operation to the tensor
	b.op.Apply_InplaceOp(A)
	return A
}

/*
* @notice Axis_ElementOperation() applies an operation to each element along a given axis of a Tensor. This results in a Tensor that
* is the same shape as the original Tensor.
* @dev Due to the nature of this problem, we can tale advantage of the code that was written for performing batched operations.
* By Transposing the axis of operation to the first axis, the operation can be treated as a batched operation. To which the result
* is Transposed back to the original shape.
 */
func (A *Tensor) AxisInplaceOperation(axis int, op InplaceOperation) *Tensor {
	if axis < 0 || axis >= len(A.Shape) {
		panic("Within AxisInplaceOperation() --- Invalid axis")
	}

	// Transpose Tensor so that the axis of operation is the first axis
	axisReordering := Permute_Shape(A.Shape, axis, 0) // Using the previously defined function
	A_Transpose := A.Permute(axisReordering)

	// Define the batch operation that applies op
	batchOp := &BatchInplaceOperation{op: op} // BatchInplaceOperation needs to be defined

	// Perform the operation along the first axis using Batch_Tensor_Tensor_Operation
	result := BatchedOperation(batchOp, A_Transpose)

	// Transpose the result back to the original shape
	resultTranspose := result.Permute(Permute_Shape(result.Shape, 0, axis))

	return resultTranspose
}

//============================================================================================================================== AxisCollapsingOperation

/*
* @notice An AxisCollapsingOperation is a Tensor operation performed along an axis that results in the axis being collapsed
* @dev The way this algorithm works is that for a given Tensor and a specified axis to perform a collapsing operation along, we will first create
* a Zero_Tensor that will contain the result. Then we will iterate through that axis, create a Partial of the Tensor on that axis and
* element using Remove_Dim(). This Partial will then be passed into a go routine that will use Mutexes to add the Partial to the result
* Tensor. This will be done in parallel for each element along the axis. The result Tensor will then be returned.
 */
func (A *Tensor) Axis_Collapsing_Operation(axis int, op Collapsing_Operation) *Tensor {
	if axis < 0 || axis >= len(A.Shape) {
		// Handle invalid axis error
		return nil
	}

	// Create a Zero_Tensor of the same shape as A but with the axis removed
	resultShape := make([]int, 0, len(A.Shape)-1)
	resultShape = append(resultShape, A.Shape[:axis]...)
	resultShape = append(resultShape, A.Shape[axis+1:]...)
	resultTensor := ZeroTensor(resultShape, false)

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
