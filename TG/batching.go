package TG


//import "fmt"

// batching.go contains a general interface for batching within any Tensor Operation.

// This interface contains different variations of the Execute method that can be used to perform batching.
// type Batching_Interface interface {
// 	Execute_Initializer(shape []int) *Tensor
// 	Execute_Tensor_Void(A *Tensor)
// 	Execute_Tensor_Tensor(A *Tensor) *Tensor
// 	Execute_TwoTensor_Tensor(A, B *Tensor) *Tensor
// 	Execute_ThreeTensor_Tensor(A, B, C *Tensor) *Tensor // Execute the operation and return a tensor.
// }

// func Batched_Operation(op Batching_Interface, tensors ...*Tensor) *Tensor {

// 	switch len(tensors) {
// 	// case 0:
// 	// 	// Handle the case where no tensors are provided, and an initializer is needed
// 	// 	if initializer, ok := op.(Batched_Initializer_Interface); ok {
// 	// 		return initializer.Execute(initializer.someShape) // someShape needs to be determined based on your requirements
// 	// 	}
// 	case 1:
// 		// Single tensor input
// 		if singleTensorOp, ok := op.(Batch_Tensor_Tensor_Interface); ok {
// 			return singleTensorOp.Execute(tensors[0])
// 		}
// 		if voidOp, ok := op.(Batch_Tensor_Void_Interface); ok {
// 			voidOp.Execute(tensors[0])
// 			return nil // or return some meaningful value
// 		}
// 	case 2:
// 		// Two tensor inputs
// 		if twoTensorOp, ok := op.(Batch_TwoTensor_Tensor_Interface); ok {
// 			return twoTensorOp.Execute(tensors[0], tensors[1])
// 		}
// 	case 3:
// 		// Three tensor inputs
// 		if threeTensorOp, ok := op.(Batch_ThreeTensor_Tensor_Interface); ok {
// 			return threeTensorOp.Execute(tensors[0], tensors[1], tensors[2])
// 		}
// 	default:
// 		panic("Unsupported number of tensors for batch operation")
// 	}

// 	return nil
// }

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

////--------------------------------------------------------------------------------------------------Single Tensor Input --- Single Tensor Output

type Batch_Tensor_Tensor_Interface interface {
	Execute(tensor *Tensor) *Tensor // Execute the operation and return a tensor.
}

// This function concurrently processes each element of a batch tensor and returns a batch tensor of the same shape.
// The operation is defined by the op parameter, which is an instance of a struct that implements the Batch_Tensor_Tensor_Interface
// interface. The function returns a pointer to the batched output tensor.
func Batch_Tensor_Tensor_Operation(op Batch_Tensor_Tensor_Interface, A *Tensor) *Tensor {
	type result struct {
		index  int
		tensor *Tensor
	}

	results := make(chan result, A.Shape[0]) // Channel to collect results
	orderMap := make(map[int]*Tensor)        // Map to track order of batch in results

	// Define the eachElementOp function
	eachElementOp := func(example int) *Tensor {
		Batched_Output := A.Remove_Dim(0, example) // Retrieve the element from the batch tensor
		Output := op.Execute(Batched_Output)
		Output.Shape = append([]int{1}, Output.Shape...) // Add a singleton dimension to the front of the shape
		return Output
	}

	// Launch a goroutine for each element in the batch
	for i := 0; i < A.Shape[0]; i++ {
		go func(index int) {
			output := eachElementOp(index)
			results <- result{index: index, tensor: output}
		}(i)
	}

	// Collect results from the channel
	for i := 0; i < A.Shape[0]; i++ {
		res := <-results
		orderMap[res.index] = res.tensor
	}
	close(results)

	// Start collecting the results into a batched tensor by retrieving the first element
	Batched_Output := orderMap[0]

	// Concatenate in the correct order
	for i := 1; i < A.Shape[0]; i++ {
		Batched_Output = Batched_Output.Concat(orderMap[i], 0)
	}

	return Batched_Output
}

//--------------------------------------------------------------------------------------------------Double Tensor Input --- Single Tensor Output

type Batch_TwoTensor_Tensor_Interface interface {
	Execute(tensor1, tensor2 *Tensor) *Tensor // Execute the operation and return a tensor.
}

// This function concurrently processes each element of a batch tensor and returns a batch tensor of the same shape.
// The operation is defined by the op parameter, which is an instance of a struct that implements the Batch_TwoTensor_Tensor_Interface
// interface. The function returns a pointer to the batched output tensor.
func Batch_TwoTensor_Tensor_Operation(op Batch_TwoTensor_Tensor_Interface, A, B *Tensor) *Tensor {
	type result struct {
		index  int
		tensor *Tensor
	}

	if A.Shape[0] != B.Shape[0] {
		panic("Tensors A and B must have the same batch size")
	}

	results := make(chan result, A.Shape[0]) // Channel to collect results
	orderMap := make(map[int]*Tensor)        // Map to track order

	// Define the eachElementOp function
	eachElementOp := func(example int) *Tensor {
		tensor1 := A.Remove_Dim(0, example) // Retrieve elements
		tensor2 := B.Remove_Dim(0, example)

		Output := op.Execute(tensor1, tensor2)           // Execute the operation on the current element
		Output.Shape = append([]int{1}, Output.Shape...) // Add a singleton dimension to the front of the shape
		return Output
	}

	// Launch a goroutine for each element in the batch
	for i := 0; i < A.Shape[0]; i++ {
		go func(index int) {
			output := eachElementOp(index)
			results <- result{index: index, tensor: output}
		}(i)
	}

	// Collect results from the channel
	for i := 0; i < A.Shape[0]; i++ {
		res := <-results
		orderMap[res.index] = res.tensor
	}
	close(results)

	// Start Batched Output Process by getting the first element
	Batched_Output := orderMap[0]

	// Concatenate in the correct order
	for i := 1; i < A.Shape[0]; i++ {
		Batched_Output = Batched_Output.Concat(orderMap[i], 0)
	}

	return Batched_Output
}

//--------------------------------------------------------------------------------------------------Three Tensor Input --- Single Tensor Output

type Batch_ThreeTensor_Tensor_Interface interface {
	Execute(tensor1, tensor2, tensor3 *Tensor) *Tensor // Execute the operation and return a tensor.
}

// This function concurrently processes each element of a batch tensor and returns a batch tensor of the same shape.
// The operation is defined by the op parameter, which is an instance of a struct that implements the Batch_TwoTensor_Tensor_Interface
// interface. The function returns a pointer to the batched output tensor.
func Batch_ThreeTensor_Tensor_Operation(op Batch_ThreeTensor_Tensor_Interface, A, B, C *Tensor) *Tensor {
	type result struct {
		index  int
		tensor *Tensor
	}

	if A.Shape[0] != B.Shape[0] {
		panic("Tensors A and B must have the same batch size")
	}

	results := make(chan result, A.Shape[0]) // Channel to collect results
	orderMap := make(map[int]*Tensor)        // Map to track order

	// Define the eachElementOp function
	eachElementOp := func(example int) *Tensor {
		tensor1 := A.Remove_Dim(0, example) // Retrieve elements
		tensor2 := B.Remove_Dim(0, example)
		tensor3 := C.Remove_Dim(0, example)

		Output := op.Execute(tensor1, tensor2, tensor3)  // Execute the operation on the current element
		Output.Shape = append([]int{1}, Output.Shape...) // Add a singleton dimension to the front of the shape
		return Output
	}

	// Launch a goroutine for each element in the batch
	for i := 0; i < A.Shape[0]; i++ {
		go func(index int) {
			output := eachElementOp(index)
			results <- result{index: index, tensor: output}
		}(i)
	}

	// Collect results from the channel
	for i := 0; i < A.Shape[0]; i++ {
		res := <-results
		orderMap[res.index] = res.tensor
	}
	close(results)

	// Start Batched Output Process by getting the first element
	Batched_Output := orderMap[0]

	// Concatenate in the correct order
	for i := 1; i < A.Shape[0]; i++ {
		Batched_Output = Batched_Output.Concat(orderMap[i], 0)
	}

	return Batched_Output
}
