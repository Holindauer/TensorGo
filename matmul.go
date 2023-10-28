package main

import (
	"fmt"
	"sync"
)

type Tensor struct {
	shape []int
	data  []float64 // <--- this 1D slice to store flattened tensor
}

func New_Tensor(shape []int) *Tensor {

	t := new(Tensor) //  <--- this is a pointer to a tensor
	t.shape = shape

	// compute the total number of elements in the tensor
	num_elements := 1
	for _, dim := range shape {
		num_elements *= dim
	}

	t.data = make([]float64, num_elements) // create slice of floats for data

	return t
}

func Matmul(A *Tensor, B *Tensor) *Tensor {

	// check if tensor shapes are compatible for matmul
	if len(A.shape) != 2 || len(B.shape) != 2 {
		panic("Tensors must both be 2D to compute matmul")
	}

	// check if mxn and nxp
	if A.shape[1] != B.shape[0] {
		panic("2D Tensors must be compatible for matmul")
	}

	C := New_Tensor([]int{A.shape[0], B.shape[1]}) // <-- returns pointer to Tensor struct

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

func computeRow(A *Tensor, B *Tensor, C *Tensor, start int, end int, wg *sync.WaitGroup) {
	defer wg.Done()

	fmt.Printf("Computing rows %d to %d\n", start, end-1)

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
