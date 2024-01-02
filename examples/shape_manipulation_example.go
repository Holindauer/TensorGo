package main

import (
	"fmt"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func main() {

	/*
		The following code contains examples of how to used Tensor-Go's shape manipulation functions
		to alter the shape of a Tensor.

		The following functions are covered:

		- Partial()
		- Reshape()
		- Transpose()
		- Concat()
		- Permute_Shape()
		- Extend_Shape()
		Extend_Dim()
		- Remove_Dim()
		- Remove_Singletons()
		- Add_Singleton()

		NOTE: Unlike many of the other functions in this library, these functions do not neccesarilly
		have a batched version.

		The
	*/

	// init a tensor of shape (5, 7, 3, 1, 2, 6, 1, 1)
	var A *Tensor = Range_Tensor([]int{5, 7, 3, 1, 2, 6, 1, 1}, false) // <--- non-batched

	fmt.Println("\nOriginal Tensor Shape: ", A.Shape)

	// Next let's add a singleton dimension to the end of the tensor
	A = A.Add_Singleton()
	fmt.Println("\nAdded Singleton Dimension: ", A.Shape)

	// Now lets remove all of the singletons with the Remove_Singletons() function
	A = A.Remove_Singletons()
	fmt.Println("\nRemoved Singletons: ", A.Shape)

	// Let's swap the second and fourth dimension by first using Permute_Shape() to get a slice representing
	// The reordering of the dimensions. Then we can use the Transpose() function to apply the reording to the
	// underlying contiguous memory.
	var reordering []int = Permute_Shape(A.Shape, 1, 3)
	A = A.Transpose(reordering)
	fmt.Println("\nAxis Reordering: ", reordering)
	fmt.Println("Swapped Second and Fourth Dimensions: ", A.Shape)

	// Let's apply a Reshape(). It's important to NOTE that the Reshape() function does not change the underlying
	// contiguous memory. It only changes the shape me of the Tensor. The product of the shape must be the same
	// as that of the original shape.
	A = A.Reshape([]int{6, 5, 42}, false) // <-- NOTE: Reshape() has batching capabilities
	fmt.Println("\nReshaped Tensor: ", A.Shape)

	// Let's concatenate a [6, 7, 42] Tensor along the 1'th dimmension
	var B *Tensor = Range_Tensor([]int{6, 7, 42}, false) // <-- non-batched
	A = A.Concat(B, 1)
	fmt.Println("\nConcatenated Tensor: ", A.Shape)

	// Let's now take a partial of the Tensor. This works the same as python's slice indexing.
	A = A.Partial(":, 1:7, 34:")
	fmt.Println("\nPartial Tensor: ", A.Shape)

	// Let's now remove a dimmension from the Tensor. To do this, we have to specify axis we
	// want to remove as well as which element from that dimension to keep values from in the
	// other remaining dimensions
	A = A.Remove_Dim(1, 5) // <-- remove the 1'th dimmension and keep the 5'th "slice" from it
	fmt.Println("\nRemoved Dimmension: ", A.Shape)

	// Now let's use the Extend_Shape() function to add a new dim to the end of the Tensor in
	// which each element of that dim is a copy of the original Tensor
	A = A.Extend_Shape(3)
	fmt.Println("\nExtended Shape: ", A.Shape)

	// Next, let's extend the first dimmension, which will add all zero elements in the newly added
	// elements of the preexisting first dimmension.
	// We are specifying that we want to extend the first dimmension by 3 elements
	A = A.Extend_Dim(0, 3)
	fmt.Println("\nExtended Dimmension: ", A.Shape)
}
