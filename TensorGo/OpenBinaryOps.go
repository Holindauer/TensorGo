package TG

/*
* @notice OpenBinaryOps.go contains functions that accept two tensors which do not have either
* the same input sizes across them or an output size that matches that of the input.
 */

//===================================================================================================================== Broadcast Variations of Closed Binary Ops

/*
* @notice the following open binary operations are variations of closed binary operations that involve broadcasting
 */

// broadcast.go contains versions of preexisting function in the format of broadcasting.

// This function broadcasts elementwise addition of the A Tensor onto the B Tensor. The Tensor of shape [n...] is treated as [1, n...] and applied as an
func (Broadcast_Arg *Tensor) Broadcast_Add(Broadcast_Onto *Tensor) *Tensor {

	// define an anon function that takes two tensors and returns a tensor after applying the sum operation
	op := func(A *Tensor, B *Tensor) *Tensor {
		return Add(A, B, false)
	}

	return Broadcast_Arg.Broadcast(Broadcast_Onto, op)
}

// This function broadcasts elementwise addition of the A Tensor onto the B Tensor. The Tensor of shape [n...] is treated as [1, n...] and applied as an
func (Broadcast_Arg *Tensor) Broadcast_Subtract(Broadcast_Onto *Tensor) *Tensor {

	// define an anon function that takes two tensors and returns a tensor after applying the sum operation
	op := func(A *Tensor, B *Tensor) *Tensor {
		return Subtract(A, B, false)
	}

	return Broadcast_Arg.Broadcast(Broadcast_Onto, op)
}

// =========================================================================================================== Scalar Multiplication

// Define a struct that implements the Batch_Tensor_Tensor_Operation interface. See batching.go for more details
type ScalarMultOp struct{ scalar float64 }

func (bsm ScalarMultOp) Execute(tensors ...*Tensor) *Tensor {

	A := tensors[0]

	// create new tensor to store result
	cA := A.Copy()
	for i := 0; i < len(A.Data); i++ {
		cA.Data[i] *= bsm.scalar
	}
	return cA
}

// This funciton performs scalar multiplication on a tensor in place
// It returns a pointer to a new tensor
func (A *Tensor) Scalar_Mult(scalar float64, batching bool) *Tensor {

	// initialize the batched op
	scalarMult := ScalarMultOp{scalar: scalar}

	if batching {
		return BatchedOperation(scalarMult, A) // batched op
	}
	return scalarMult.Execute(A) // single op
}

// =========================================================================================================== Argmax

// // Define a struct that implements the Batch_Tensor_Tensor_Operation interface. See batching.go for more details
// type ArgmaxOp struct{}

// func (argmax ArgmaxOp) Execute(tensors ...*Tensor) *Tensor {

// 	A := tensors[0]

// 	// create new tensor to store result
// 	cA := A.Copy()
// 	for i := 0; i < len(A.Data); i++ {
// 		cA.Data[i] = float64(cA.Argmax())
// 	}
// 	return cA
// }

// // This funciton performs argmax on a tensor in place
// // It returns a pointer to a new tensor
// func (A *Tensor) Argmax(batching bool) *Tensor {

// 	// initialize the batched op
// 	argmax := ArgmaxOp{}

// 	if batching {
// 		return BatchedOperation(argmax, A) // batched op
// 	}
// 	return argmax.Execute(A) // single op
// }
