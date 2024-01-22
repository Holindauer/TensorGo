package TG

// matrix.go contains functions for manipulating 2D

import (
	"fmt"
)

//===================================================================================================================== Matmul()

type MatMulOp struct{}

// Implementing the Execute method of IBatching interface
func (op MatMulOp) Execute(tensors ...*Tensor) *Tensor {
	// Assumes tensors length will be 2 for matrix multiplication
	A, B := tensors[0], tensors[1]

	// Address Matrix Vector Multiplication
	if len(B.Shape) == 1 {
		B = B.Add_Singleton(0)
	}

	// Check that the two Tensors are compatible for matrix multiplication
	Check_MatMul_Compatibility(A, B)

	C := ZeroTensor([]int{A.Shape[0], B.Shape[1]}, false)
	var sum float64

	for row := 0; row < C.Shape[0]; row++ {
		for col := 0; col < C.Shape[1]; col++ {
			sum = 0
			for k := 0; k < A.Shape[1]; k++ {
				sum += A.Get([]int{row, k}) * B.Get([]int{k, col})
			}

			C.Data[C.Index([]int{row, col})] = sum
		}
	}

	return C
}

func MatMul(A *Tensor, B *Tensor, batching bool) *Tensor {

	matmul := MatMulOp{} // Create an instance of Batched_Matmul

	if batching {
		// If batching is true, call BatchedOperation directly
		return BatchedOperation(matmul, A, B)
	} else {
		// If batching is false, call the Execute method directly
		return matmul.Execute(A, B)
	}
}

//===================================================================================================================== Gradient Tracked Matrix Mulitplication

type MatMulGradOp struct{}

// Implementing the Execute method of IBatching interface
func (op MatMulGradOp) Execute(tensors ...*Tensor) *Tensor {
	// Assumes tensors length will be 2 for matrix multiplication
	A, B := tensors[0], tensors[1]

	// In case of Matrix Vector Multiplication
	if len(B.Shape) == 1 {
		B = B.Add_Singleton(0)
	}
	if len(A.Shape) == 1 {
		A = A.Add_Singleton(0)
	}

	// Check that the two Tensors are compatible for matrix multiplication
	Check_MatMul_Compatibility(A, B)

	C := ZeroTensor([]int{A.Shape[0], B.Shape[1]}, false)
	C.DataReqGrad = make([]*Value, len(C.Data))
	var sum *Value

	for row := 0; row < C.Shape[0]; row++ {
		for col := 0; col < C.Shape[1]; col++ {

			sum = NewValue(0.0, nil, "")

			for k := 0; k < A.Shape[1]; k++ {

				// Gradient Tracked Dot Product
				elementA := A.DataReqGrad[TheoreticalIndex([]int{row, k}, A.Shape)]
				elementB := B.DataReqGrad[TheoreticalIndex([]int{k, col}, B.Shape)]

				// grad tracked multiplication
				var mul *Value = elementA.Mul(elementB)

				sum = sum.Add(mul)
			}

			C.DataReqGrad[C.Index([]int{row, col})] = sum
		}
	}

	return C
}

func MatMulGrad(A *Tensor, B *Tensor, batching bool) *Tensor {

	matmul := MatMulGradOp{} // Create an instance of Batched_Matmul

	if batching {

		// If batching is true, call BatchedOperation directly
		return BatchedOperation(matmul, A, B)
	} else {
		// If batching is false, call the Execute method directly
		return matmul.Execute(A, B)
	}
}

//===================================================================================================================== Display_Matrix()

// TODO <--- implement batching for non returning functions into the new BatchedOperation abstraction

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

//===================================================================================================================== Augment_Matrix()

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
