package main

import (
	"fmt"

	. "github.com/Holindauer/Tensor-Go/TG"
)

/*
	This examples demonstrates how to initialize the different types of Tensors using Tensor-Go.

	As a general rule, most functionsin Tensor-Go have a batched and non-batched version. It there is a batched
	version of a function, it will be controlled by the last argument, which is a boolean.
*/

func main() {

	// Ones Tensor
	var ones *Tensor = Ones_Tensor([]int{3, 4}, false) // non-batched
	fmt.Println("\nOnes Tensor: ")
	Display_Matrix(ones, false)

	// Zeros Tensor
	var zeros *Tensor = Zero_Tensor([]int{2, 3, 4}, true) // batched
	fmt.Println("\nZeros Tensor: ")
	Display_Matrix(zeros, true)

	// Range Tensor
	var range_tensor *Tensor = Range_Tensor([]int{2, 3, 4}, true) // batched
	fmt.Println("\nRange Tensor: ")
	Display_Matrix(range_tensor, true)

	// Constant Tensor
	var const_tensor *Tensor = Const_Tensor([]int{2, 3, 4}, 5, true) // batched
	fmt.Println("\nConstant Tensor: ")
	Display_Matrix(const_tensor, true)

	// Random Tensor
	var random_tensor *Tensor = RandFloat64_Tensor([]int{2, 3, 4, 8, 6}, 0, 1, true) // batched
	fmt.Println("\nRandom Tensor Shape", random_tensor.Shape)

	// Copy
	var copy_tensor *Tensor = random_tensor.Copy()
	fmt.Println("\nCopy Tensor Shape", copy_tensor.Shape)

	// Identity
	var identity_tensor *Tensor = Eye([]int{4, 4}, false) // non-batched
	fmt.Println("\nIdentity Tensor: ")
	Display_Matrix(identity_tensor, false)
}
