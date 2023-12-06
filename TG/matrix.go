package TG


// matrix.go contains functions for manipulating 2D

import (
	"fmt"
)

//-------------------------------------------------------------------------------------------------------------- Matmul()

type Batched_Matmul struct{}

// This method of the Batch_Matmul struct computes the matrix multiplication of two 2D tensors
func (op Batched_Matmul) Execute(A *Tensor, B *Tensor) *Tensor {

	// Address Matrix Vector Multiplication
	if len(B.Shape) == 1 {
		B = B.Add_Singleton()
	}

	// Check that the two Tensors are compatible for matrix multiplication
	Check_MatMul_Compatibility(A, B)

	C := Zero_Tensor([]int{A.Shape[0], B.Shape[1]}, false) // <-- returns pointer to Tensor struct
	var sum float64

	for row := 0; row < C.Shape[0]; row++ { // <-- iterate through rows of C

		for col := 0; col < C.Shape[1]; col++ { // <-- iterate through columns of C
			sum = 0                           // <-- reset sum
			for k := 0; k < A.Shape[1]; k++ { // compute dot product of row of A and column of B
				sum += A.Retrieve([]int{row, k}) * B.Retrieve([]int{k, col})
			}

			// write to C.data slice
			C.Data[C.Index([]int{row, col})] = sum
		}
	}

	return C
}

// This function computes the dot product of two vectors
func MatMul(A *Tensor, B *Tensor, batching bool) *Tensor {

	if batching {
		return Batch_TwoTensor_Tensor_Operation(Batched_Matmul{}, A, B)
	} 
	return Batched_Matmul{}.Execute(A, B)
	
}

//-------------------------------------------------------------------------------------------------------------- Display_Matrix()

type Batched_Display_Matrix struct{}

// This method of the Batched_Display_Matrix struct displays a 2D tensor as a matrix
func (op Batched_Display_Matrix) Execute(A *Tensor) {
	fmt.Println()
	if len(A.Shape) == 2 {
		// Handling 2D matrix
		for i := 0; i < A.Shape[0]; i++ {
			for j := 0; j < A.Shape[1]; j++ {
				fmt.Printf("%v ", A.Data[i*A.Shape[1]+j])
			}
			fmt.Println()
		}
	} else if len(A.Shape) == 1 {
		// Handling vector
		for i := 0; i < A.Shape[0]; i++ {
			fmt.Printf("%v ", A.Data[i])
		}
		fmt.Println()
	} else {
		fmt.Println("Within Display_Matrix(): Tensor must be 1D or 2D to display as matrix or vector")
	}
}

// this function is used to display a 2D tensor as a matrix
func Display_Matrix(A *Tensor, batching bool) {
	if !batching {
		Batched_Display_Matrix{}.Execute(A)
	} else {
		Batch_Tensor_Void_Operation(Batched_Display_Matrix{}, A)
	}

}

//-------------------------------------------------------------------------------------------------------------- Augment_Matrix()

// This function creates an augmented matrix fromt two matrix (2D) Tensors for use in the Gaussian_Elimination function.
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
