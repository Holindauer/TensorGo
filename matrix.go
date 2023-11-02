package main

// This file contains functions for manipulating 2D

import (
	"fmt"
	"sync"
)

// This function computes the dot product of two vectors
func Matmul(A *Tensor, B *Tensor) *Tensor {

	// check if tensor shapes are compatible for matmul
	if len(A.shape) != 2 || len(B.shape) != 2 {
		panic("Tensors must both be 2D to compute matmul")
	}

	// check if mxn and nxp
	if A.shape[1] != B.shape[0] {
		panic("2D Tensors must be compatible for matmul")
	}

	C := Zero_Tensor([]int{A.shape[0], B.shape[1]}) // <-- returns pointer to Tensor struct

	numGoroutines := 4
	chunkSize := C.shape[0] / numGoroutines

	// because each index of C is indepentent of the other, we will write directly to the
	// C.data slice within the C tensor, and there is no need for a mutex.

	var wg sync.WaitGroup

	for i := 0; i < numGoroutines; i++ {

		wg.Add(1) // Increment the WaitGroup counter

		start := i * chunkSize //  compute bounds of the chunk
		end := start + chunkSize

		if i == numGoroutines-1 {
			end = C.shape[0] // Ensure the last chunk includes any remaining elements
		}

		go computeRow(A, B, C, start, end, &wg)
	}
	return C
}

// This is a helper function for Matmul() above. It computes the dot product of a chunk of the vectors
func computeRow(A *Tensor, B *Tensor, C *Tensor, start int, end int, wg *sync.WaitGroup) {
	defer wg.Done()

	for row := start; row < end; row++ { // <-- iterate through rows of C

		for col := 0; col < C.shape[1]; col++ { // <-- iterate through columns of C

			var sum float64
			for k := 0; k < A.shape[1]; k++ { // compute dot product of row of A and column of B
				A_idx := Index([]int{row, k}, A.shape)
				B_idx := Index([]int{k, col}, B.shape)

				sum += A.data[A_idx] * B.data[B_idx]
			}
			// compute flat index of C
			C_idx := Index([]int{row, col}, C.shape)

			// write to C.data slice directly
			C.data[C_idx] = sum
		}
	}
}

// this function is used to display a 2D tensor as a matrix
func Display_Matrix(t *Tensor) {
	if len(t.shape) == 2 {
		// Handling 2D matrix
		for i := 0; i < t.shape[0]; i++ {
			for j := 0; j < t.shape[1]; j++ {
				fmt.Printf("%v ", t.data[i*t.shape[1]+j])
			}
			fmt.Println()
		}
	} else if len(t.shape) == 1 {
		// Handling vector
		for i := 0; i < t.shape[0]; i++ {
			fmt.Printf("%v ", t.data[i])
		}
		fmt.Println()
	} else {
		fmt.Println("Tensor must be 1D or 2D to display as matrix or vector")
	}
}
