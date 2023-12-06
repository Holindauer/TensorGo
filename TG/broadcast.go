package TG


// broadcast.go contains versions of preexisting function in the format of broadcasting.

// To broadcast a Tensor of shape[n...] onto [m, n...]. The Tensor of shape is treated as [1, n...] and applied as an
// elementwise operation along each element along the 0'th axis of [m, n...]. In this function Broadcast_Arg is the Tensor
// of shape [n...] and Broadcast_To is the Tensor of shape [m, n...]. The result is a Tensor of shape [m, n...].
//
// The op function is a function that should take two Tensors of shape [1, n...] and return a Tensor of shape [1, n...]. This operation
// will perform what ever function was intended to be broadcasted onto B.
func (Broadcast_Arg *Tensor) Broadcast(Broadcast_Onto *Tensor, op func(A *Tensor, B *Tensor) *Tensor) *Tensor {

	if !isEqual(Broadcast_Arg.Shape, Broadcast_Onto.Shape[1:]) {
		panic("Broadcast() requires that the shape of the first Tensor be equal to the shape of the second Tensor with the first axis removed")
	}

	// Add a singleton to the beginning of the shape of Broadcast_Arg so that it can be concatenated upon
	Broadcast_Arg_Copy := Broadcast_Arg.Copy() // we are copying the Tensor so we don't want to modify the original
	Broadcast_Arg_Copy.Shape = append([]int{1}, Broadcast_Arg.Shape...)

	// For the first element along the 0'th axis of Broadcast_Onto, apply the operation
	result := Broadcast_Onto.Remove_Dim(0, 0).Reshape(Broadcast_Arg_Copy.Shape, false)
	result = op(result, Broadcast_Arg_Copy) // apply the operation to the first element along the 0'th axis

	for i := 1; i < Broadcast_Onto.Shape[0]; i++ {

		elementResult := Broadcast_Onto.Remove_Dim(0, i).Reshape(Broadcast_Arg_Copy.Shape, false)

		result = result.Concat(op(elementResult, Broadcast_Arg_Copy), 0)
	}
	return result
}

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
