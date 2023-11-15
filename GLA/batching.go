package GLA

//import "fmt"

// This source file contains a general interface for batching within any Tensor Operation.

// -------------------------------------------------------------------------------------------------- Batched Initialization

type Batched_Initializer_Interface interface {
	Execute(shape []int) *Tensor // Execute the operation and return a tensor.
}

func Batched_Initializer_Operation(op Batched_Initializer_Interface, shape []int) *Tensor {

	// Determine the shape of each element (remove the first dimension from the shape)
	elementShape := shape[1:]

	// Anon Function for creating each element of the batch
	eachElementOp := func(example int) *Tensor {

		// Execute the operation on the element
		Output := op.Execute(elementShape)

		// Add a singleton dimension to the front of the shape
		Output.Shape = append([]int{1}, elementShape...)
		return Output

	}

	Batched_Tensor := eachElementOp(0) // <--- Execute the operation on the first element

	for i := 1; i < shape[0]; i++ { // <--- num batch elements
		Batch_Element := eachElementOp(i)
		Batched_Tensor = Batched_Tensor.Concat(Batch_Element, 0)
	}

	return Batched_Tensor
}

// --------------------------------------------------------------------------------------------------Single Tensor Input --- Void Output

type Batch_Tensor_Void_Interface interface {
	Execute(tensor *Tensor)
}

func Batch_Tensor_Void_Operation(op Batch_Tensor_Void_Interface, A *Tensor) {

	//define the above code as an anon func
	eachElementOp := func(example int) {
		Batched_Output := A.Remove_Dim(0, example) // <--- retrieve the first element from the 0'th dim of the batch tensor
		op.Execute(Batched_Output)
	}

	// Start Batched Output Process by executing the operation on the first element
	eachElementOp(0)
	for i := 1; i < A.Shape[0]; i++ { // <--- iterate through the remaining elements of the batch tensor
		eachElementOp(i) // <--- execute the operation on the current element
	}
}

////--------------------------------------------------------------------------------------------------Single Tensor Input --- Single Tensor Output

type Batch_Tensor_Tensor_Interface interface {
	Execute(tensor *Tensor) *Tensor // Execute the operation and return a tensor.
}

func Batch_Tensor_Tensor_Operation(op Batch_Tensor_Tensor_Interface, A *Tensor) *Tensor {

	//define the above code as an anon func
	eachElementOp := func(example int) *Tensor {
		Batched_Output := A.Remove_Dim(0, example) // <--- retrieve the first element from the 0'th dim of the batch tensor
		Output := op.Execute(Batched_Output)
		Output.Shape = append([]int{1}, Output.Shape...) // <--- add a singleton dimension to the front of the shape
		return Output
	}

	// Start Batched Output Process by executing the operation on the first element
	Batched_Output := eachElementOp(0)

	for i := 1; i < A.Shape[0]; i++ { // <--- iterate through the remaining elements of the batch tensor
		Output := eachElementOp(i) // <--- execute the operation on the current element
		Batched_Output = Batched_Output.Concat(Output, 0)
	}

	return Batched_Output
}

//--------------------------------------------------------------------------------------------------Double Tensor Input --- Single Tensor Output   // Eventually merge all batch functions into the same func

type Batch_TwoTensor_Tensor_Interface interface {
	Execute(tensor1, tensor2 *Tensor) *Tensor // Execute the operation and return a tensor.
}

func Batch_TwoTensor_Tensor_Operation(op Batch_TwoTensor_Tensor_Interface, A, B *Tensor) *Tensor {

	//define the above code as an anon func
	eachElementOp := func(example int) *Tensor {
		tensor1 := A.Remove_Dim(0, example) // <--- retrieve elements
		tensor2 := B.Remove_Dim(0, example)

		Output := op.Execute(tensor1, tensor2)           // <--- execute the operation on the current element
		Output.Shape = append([]int{1}, Output.Shape...) // <--- add a singleton dimension to the front of the shape
		return Output
	}

	// Start Batched Output Process by executing the operation on the first element
	Batched_Output := eachElementOp(0)
	for i := 1; i < A.Shape[0]; i++ { // <--- iterate through the remaining elements of the batch tensor
		Output := eachElementOp(i) // <--- execute the operation on the current element
		Batched_Output = Batched_Output.Concat(Output, 0)
	}

	return Batched_Output
}
