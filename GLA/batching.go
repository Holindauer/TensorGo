package GLA

// This source file contains a general interface for batching within any Tensor Operation.

// The basic idea is that for a function that performs an operation on a Tensor will include within its parameter list a boolean
// on whether batching should be performed. Before the operation has been conducted, an anonymous function that perfrorms its task
// on a single Tensor will be created. Following this will be a conditional statement that checks the batching boolean. This anon
// function, its parameters, and the batching boolean will placed into an interface and passed into the batch function.

// The batch() function will employ a conditional statement that checks the batching boolean. If batching is false, then the anon
// function willjust run the operation on the single Tensor. If batching is true, then the anon function use the Remove_Dim() function
// to perform the operation on each Tensor in the batch. Each of these Tensors will be concatenationg back into their batch and a
// pointer to this batch will be returned.

// I think that depening on the type of operation, the specific implementation of batching may be different. I started out with trying to implement
// the vector dot product, but I think that the implementation of batching for this operation is different than for other operations. This is
// due to input type and amount beiong input affecting how the logic of the generalization of batching is implemented. Some of the variations are

// ops that act on a single Tensor and return a single Tensor
// ops that act on a single Tensor and return a float64

// ops that act on two Tensors and return a single Tensor
// ops that act on two Tensors and return a float64

// These will need to be addrssed individually.

// The following functions are named using the follwoing naming scheme: Batch_<input type/amount>_<output type>

// This interface is used to generalize batching within any Tensor Operation. It is specific to operations that take in
// a single Tensor Argument and return a single Tensor.

// -------------------------------------------------------------------------------------------------- Batched Initialization

type Batched_Initializer_Interface interface {
	Execute(shape []int) *Tensor // Execute the operation and return a tensor.
}

func Batched_Initializer_Operation(op Batched_Initializer_Interface, shape []int) *Tensor {
	// Determine the shape of each element (remove the first dimension from the shape)
	elementShape := shape[1:]

	// Anon Function for creating each element of the batch
	eachElementOp := func(example int) *Tensor {
		Batch_Element := op.Execute(elementShape).Add_Singleton() // <--- create element of batch with singleton dim added
		singletonReordering := Indicies_First_Last_Swapped(len(Batch_Element.Shape))
		return Batch_Element.Transpose(singletonReordering) // <--- reorder conitguous memory
	}

	Batched_Tensor := eachElementOp(0)

	for i := 1; i < shape[0]; i++ { // <--- num batch elements
		Batch_Element := eachElementOp(i)
		Batched_Tensor = Batched_Tensor.Concat(Batch_Element, 0)
	}

	return Batched_Tensor

}

//--------------------------------------------------------------------------------------------------Single Tensor Input --- Single Tensor Output

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

//--------------------------------------------------------------------------------------------------Double Tensor Input --- Single Tensor Output   // Eventually merge all batch functions into the same func

type Batch_TwoTensor_Tensor_Interface interface {
	Execute(tensor1, tensor2 *Tensor) *Tensor // Execute the operation and return a tensor.
}

func Batch_TwoTensor_Tensor_Operation(op Batch_TwoTensor_Tensor_Interface, A, B *Tensor) *Tensor {

	//define the above code as an anon func
	eachElementOp := func(example int) *Tensor {
		Batched_Output := A.Remove_Dim(example, 0)                            // <--- retrieve the first element from the 0'th dim of the batch tensor
		Output := op.Execute(Batched_Output, B).Add_Singleton()               // <--- execute the operation on the first element
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

//--------------------------------------------------------------------------------------------------Double Tensor Input --- Single Tensor Output
