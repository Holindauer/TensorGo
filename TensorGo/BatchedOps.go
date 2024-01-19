package TG

/*
* @notice batching.go contains a general interface for batching within any Tensor Operation.
 */

// @dev IBatching is an interface that conains an operation to be applied to each element of a batched tensor
type IBatching interface {
	Execute(tensors ...*Tensor) *Tensor // Closed Unary/Binary/Ternary Operations
}

func BatchedOperation(op IBatching, tensors ...*Tensor) *Tensor {
	// Ensure all tensors have the same batch size
	batchsize := tensors[0].Shape[0]
	for _, tensor := range tensors {
		if tensor.Shape[0] != batchsize {
			panic("All tensors must have the same batch size")
		}
	}

	// Batchify the provided operation
	batchedOp := Batchify(op, tensors...)

	// Define a struct to store the results of the batched operation
	type result struct {
		index  int
		tensor *Tensor
	}

	// setup channel to collect results
	results := make(chan result, batchsize)

	// setup map to track order of batch in results The Tensor is split into individual elements and processed concurrently
	// so to group the results back into a batched Tensor, we need to track the order of the elements in the batch
	orderMap := make(map[int]*Tensor)

	// Launch a goroutine for each element in the batch
	for i := 0; i < batchsize; i++ {
		go func(index int) {
			output := batchedOp(index)
			results <- result{index: index, tensor: output}
		}(i)
	}

	// Collect results from the channel (out of order)
	for i := 0; i < batchsize; i++ {
		res := <-results
		orderMap[res.index] = res.tensor
	}
	close(results)

	// Combine results in the tracked order
	Batched_Output := orderMap[0]
	for i := 1; i < batchsize; i++ {
		Batched_Output = Batched_Output.Concat(orderMap[i], 0)
	}

	return Batched_Output
}

/*
* @notice Batchify is used to determine which type of operation (unary, binary, ternary) is being performed based on the number of Tensor inputs.
* It then converts the IBatching interface into a function that can be applied to individual element of a batched Tensor.
 */
func Batchify(op IBatching, tensors ...*Tensor) func(int) *Tensor {
	switch len(tensors) {
	case 1: // Unary Operation
		A := tensors[0]
		return func(example int) *Tensor {
			Batched_Output := A.Remove_Dim(0, example)
			Output := op.Execute(Batched_Output)
			Output.Shape = append([]int{1}, Output.Shape...)
			return Output
		}

	case 2: // Binary Operation
		A, B := tensors[0], tensors[1]
		return func(example int) *Tensor {
			tensor1 := A.Remove_Dim(0, example)
			tensor2 := B.Remove_Dim(0, example)
			Output := op.Execute(tensor1, tensor2)
			Output.Shape = append([]int{1}, Output.Shape...)
			return Output
		}

	case 3: // Ternary Operation
		A, B, C := tensors[0], tensors[1], tensors[2]
		return func(example int) *Tensor {
			tensor1 := A.Remove_Dim(0, example)
			tensor2 := B.Remove_Dim(0, example)
			tensor3 := C.Remove_Dim(0, example)
			Output := op.Execute(tensor1, tensor2, tensor3)
			Output.Shape = append([]int{1}, Output.Shape...)
			return Output
		}

	default:
		panic("Unsupported number of tensors")
	}
}

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

// NOTE: At the moment, the only function that uses this interface is Display_Matrix(). I don' think this requires
//       concurrency. If Tensor to Void dunctions are added that would benefit from concurrency, the probably what
//       should be done would be to add a boolean parameter to this function that indicates whether or not to use
//       concurrency.

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
