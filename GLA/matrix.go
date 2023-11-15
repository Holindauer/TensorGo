package GLA

// This file contains functions for manipulating 2D

import (
	"fmt"
	"sync"
)

// This function computes the dot product of two vectors
func Matmul(A *Tensor, B *Tensor) *Tensor {

	// check if tensor shapes are compatible for matmul
	if len(A.Shape) != 2 || len(B.Shape) != 2 {
		panic("Within Matmul(): Tensors must both be 2D to compute matmul")
	}

	// check if mxn and nxp
	if A.Shape[1] != B.Shape[0] {
		panic("Within Matmul(): 2D Tensors must be compatible for matmul")
	}

	C := Zero_Tensor([]int{A.Shape[0], B.Shape[1]}, false) // <-- returns pointer to Tensor struct

	numGoroutines := 4
	chunkSize := C.Shape[0] / numGoroutines

	// because each index of C is indepentent of the other, we will write directly to the
	// C.data slice within the C tensor, and there is no need for a mutex.

	var wg sync.WaitGroup

	for i := 0; i < numGoroutines; i++ {

		wg.Add(1) // Increment the WaitGroup counter

		start := i * chunkSize //  compute bounds of the chunk
		end := start + chunkSize

		if i == numGoroutines-1 {
			end = C.Shape[0] // Ensure the last chunk includes any remaining elements
		}

		go computeRow(A, B, C, start, end, &wg)
	}
	return C
}

// This is a helper function for Matmul() above. It computes the dot product of a chunk of the vectors
func computeRow(A *Tensor, B *Tensor, C *Tensor, start int, end int, wg *sync.WaitGroup) {
	defer wg.Done()

	for row := start; row < end; row++ { // <-- iterate through rows of C

		for col := 0; col < C.Shape[1]; col++ { // <-- iterate through columns of C

			var sum float64
			for k := 0; k < A.Shape[1]; k++ { // compute dot product of row of A and column of B
				A_idx := Index([]int{row, k}, A.Shape)
				B_idx := Index([]int{k, col}, B.Shape)

				sum += A.Data[A_idx] * B.Data[B_idx]
			}
			// compute flat index of C
			C_idx := Index([]int{row, col}, C.Shape)

			// write to C.data slice directly
			C.Data[C_idx] = sum
		}
	}
}

// this function is used to display a 2D tensor as a matrix
func Display_Matrix(t *Tensor) {
	if len(t.Shape) == 2 {
		// Handling 2D matrix
		for i := 0; i < t.Shape[0]; i++ {
			for j := 0; j < t.Shape[1]; j++ {
				fmt.Printf("%v ", t.Data[i*t.Shape[1]+j])
			}
			fmt.Println()
		}
	} else if len(t.Shape) == 1 {
		// Handling vector
		for i := 0; i < t.Shape[0]; i++ {
			fmt.Printf("%v ", t.Data[i])
		}
		fmt.Println()
	} else {
		fmt.Println("Within Display_Matrix(): Tensor must be 1D or 2D to display as matrix or vector")
	}
}

// This fucntion creates an augmented matrix fromt two matrix (2D) Tensors for use in the Gaussian_Elimination function.
// Put simply, this fucniton checks that the two matricies are compatible for contatination alogn the 1'th axis, are 2
// dimensional, and then concatenates them along that 1'th axis.
func Augment_Matrix(A *Tensor, B *Tensor) *Tensor {

	// Check that hte two Tensors are 2 D
	if len(A.Shape) != 2 || len(B.Shape) != 2 {
		panic(" Augment_Matrix() --- Both Tensors must be 2 dimensional")
	}

	// Check that the 1'th dimmension of the two Tensors are the same
	if A.Shape[0] != B.Shape[0] {
		panic("Augment_Matrix() Both Tensors must have the same number of rows")
	}

	return A.Concat(B, 1) // <--- return the concatenation of the two Tensors along the 1'th axis
}
