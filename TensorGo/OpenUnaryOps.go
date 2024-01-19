package TG

/*
* OpenUnaryOps.go contains functions that accept a single tensor and return a value that is not the same shape as the input
 */

import (
	"math"
)

//============================================================================================================================== Summation of All Elements

// SumAllOperation represents a summation operation over the entire tensor.
type SumAllOperation struct{}

// Apply performs the summation on a chunk of the tensor's data for a go routine.
func (s SumAllOperation) Apply(A *Tensor, start, end int) float64 {
	var sum float64
	for i := start; i < end; i++ { // sum the elements in the goroutine chunk
		sum += A.Data[i]
	}
	return sum
}

// CombineResults combines the summation results from each chunk of a go routine.
func (s SumAllOperation) CombineResults(results []float64) float64 {
	var sum float64
	for _, v := range results { // sum the results from each goroutine chunk
		sum += v
	}
	return sum
}

// Sum_All() calculates the sum of all elements in a tensor. It accepts a Tensor pointer and returns a float64.
func (A *Tensor) Sum_All() float64 {
	sumOp := SumAllOperation{}       // <--- create sum operation
	return A.ScalarCollapseOp(sumOp) // <--- apply sum operation to all elements
}

//============================================================================================================================== Mean of All Elements

// MeanAllOperation represents a mean calculation operation over the entire tensor.
type MeanAllOperation struct{}

// Apply performs the mean calculation on a chunk of the tensor's data for a go routine.
func (m MeanAllOperation) Apply(A *Tensor, start, end int) float64 {
	sumOp := SumAllOperation{}        // <--- create sum operation
	sum := sumOp.Apply(A, start, end) // <--- sum goroutine chunk
	return sum / float64(end-start)   // <--- avg of chunk
}

// CombineResults combines the mean results from all chunks of a go routine.
func (m MeanAllOperation) CombineResults(results []float64) float64 {
	sumOp := SumAllOperation{}           // <-- create sum operation
	sum := sumOp.CombineResults(results) // <-- combine sum results from all chunks
	return sum / float64(len(results))   // <-- return avg of sum
}

func (A *Tensor) Mean_All() float64 {
	meanOp := MeanAllOperation{}      // <-- create mean operation
	return A.ScalarCollapseOp(meanOp) // <-- apply mean operation to all elements
}

//============================================================================================================================== Variance of All Elements

// VarAllOperation represents a variance calculation operation over the entire tensor.
type VarAllOperation struct {
	mean float64
}

// Apply is a method of VarAllOperation that performs the variance calculation on a chunk of the tensor's data for a go routine.
func (v VarAllOperation) Apply(t *Tensor, start, end int) float64 {
	var variance float64
	for i := start; i < end; i++ {
		diff := t.Data[i] - v.mean // <--- var definition: sum((x - mean)^2) / n
		variance += diff * diff    // <--- Appy() performs: (x - mean)^2 for each x
	}
	return variance
}

// CombineResults combines the variance results from all chunks.
func (v VarAllOperation) CombineResults(results []float64) float64 {
	sumOp := SumAllOperation{}           // <-- create sum operation
	sum := sumOp.CombineResults(results) // <-- combine sum results from all chunks
	return sum / float64(len(results))   // <-- return avg of sum
}

func (A *Tensor) Var_All() float64 {
	mean := A.Mean_All()                 // <-- calculate mean
	varOp := VarAllOperation{mean: mean} // <-- pass mean to variance operation
	return A.ScalarCollapseOp(varOp)     // <-- apply variance operation to all elements
}

//============================================================================================================================== Standard Deviation of All Elements

func (A *Tensor) Std_All() float64 {
	varOp := VarAllOperation{mean: A.Mean_All()} // <-- pass mean to variance operation
	variance := A.ScalarCollapseOp(varOp)        // <-- calculate variance of all elements
	return math.Sqrt(variance)                   // <-- return sqrt(variance)
}

//============================================================================================================================== Sum_Axis()

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

//============================================================================================================================== Mean_Axis()

// Mean_Axis calculates the mean of tensor elements along a specified axis.
func (A *Tensor) Mean_Axis(axis int, batching bool) *Tensor {
	sumTensor := A.Sum_Axis(axis, batching)
	count := A.Shape[axis]
	for i := range sumTensor.Data {
		sumTensor.Data[i] /= float64(count)
	}
	return sumTensor
}

//==============================================================================================================================  Var_Axis()

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

//==============================================================================================================================  Std_Axis()

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
