package main

import (
	"math"
)

// This file contains algorithms for solving systems of linear equations

// Gaussian Elimination

// Plan for implementing Gaussian Elimination:

// I will need to implement a concat() function that concatenates two tensors along a specified axis.
// This will be used to build out the upper triangular matrix while keeping the original matrix intact.
// I will also need to implement a swap() function that swaps two rows of a matrix for cases where the
// pivot is zero. This will be used to swap rows to avoid dividing by zero.
// I'll also need to figure out some kind of simple equation solver for back substitution.

// --------------------------------------------------------------------------------------------------Normal Gaussian Elimination

// Gaussian_Elimination() performs Gaussian Elimination on a system of linear equations.
// It takes two parameters: A and b, where A is an n x n matrix and b is an n x 1 column vector.
// It returns a column vector x that is the solution to the system of linear equations.
func Gaussian_Elimination(A *Tensor, b *Tensor) *Tensor {

	// If b is 1 dimmension, add a singleton dim for Augment_Matrix()
	if len(b.shape) == 1 {
		b = b.Add_Singleton()
	}

	// Augment matrix A with column vector b to form an n x (n+1) matrix Ab
	Ab := Augment_Matrix(A, b)

	// Forward Elimination
	for k := 0; k < A.shape[0]-1; k++ { // <---  iterate through first elements of each row

		// Find the k-th pivot
		i_max := Find_Pivot(Ab, k) // <--- find the row with the abs maximum element in column k
		if Ab.data[i_max*Ab.shape[1]+k] == 0 {
			panic("Gaussian_Elimination() --- No unique solution")
		}
		Ab.Swap_Rows(k, i_max) // <--- swap the k-th row with the row with the max element in column k

		// Make all rows below this one 0 in current column
		for i := k + 1; i < A.shape[0]; i++ {
			f := Ab.data[i*Ab.shape[1]+k] / Ab.data[k*Ab.shape[1]+k] // <--- f = Ab[i, k] / Ab[k, k]

			// Subtract (f * k-th row) from i-th row
			for j := k; j < A.shape[0]+1; j++ {
				Ab.data[i*Ab.shape[1]+j] = Ab.data[i*Ab.shape[1]+j] - Ab.data[k*Ab.shape[1]+j]*f
			}
		}
	}

	// Back Substitution
	x := Zero_Tensor([]int{A.shape[0], 1})
	for i := A.shape[0] - 1; i >= 0; i-- {
		sum := 0.0
		for j := i + 1; j < A.shape[0]; j++ {
			sum += Ab.data[i*Ab.shape[1]+j] * x.data[j]
		}
		x.data[i] = (Ab.data[i*Ab.shape[1]+A.shape[0]] - sum) / Ab.data[i*Ab.shape[1]+i]
	}

	return x
}

// This fucntion creates an augmented matrix fromt two matrix (2D) Tensors for use int he Gaussian_Elimination function.
// Put simply, this fucniton checks that the two matricies are compatible for contatination alogn the 1'th axis, are 2
// dimensional, and then concatenates them along that 1'th axis.
func Augment_Matrix(A *Tensor, B *Tensor) *Tensor {

	// Check that hte two Tensors are 2 D
	if len(A.shape) != 2 || len(B.shape) != 2 {
		panic(" Augment_Matrix() --- Both Tensors must be 2 dimensional")
	}

	// Check that the 1'th dimmension of the two Tensors are the same
	if A.shape[0] != B.shape[0] {
		panic("Augment_Matrix() Both Tensors must have the same number of rows")
	}

	return A.Concat(B, 1) // <--- return the concatenation of the two Tensors along the 1'th axis
}

func Find_Pivot(A *Tensor, k int) int {

	// Find the row with the maximum element in column k
	i_max := k
	max_val := 0.0
	for i := k; i < A.shape[0]; i++ {
		if math.Abs(A.data[i*A.shape[1]+k]) > max_val {
			i_max = i
			max_val = math.Abs(A.data[i*A.shape[1]+k])
		}
	}
	return i_max
}

func (A *Tensor) Swap_Rows(i int, j int) {

	// Swap the i-th and j-th rows of A
	temp := A.Get_Row(i).Copy()
	A.Set_Row(i, A.Get_Row(j))
	A.Set_Row(j, temp)
}

func (A *Tensor) Get_Row(row int) *Tensor {

	// Create a new Tensor to store the row
	B := Zero_Tensor([]int{1, A.shape[1]})

	// Copy the row into the new Tensor
	for i := 0; i < A.shape[1]; i++ {
		B.data[i] = A.data[row*A.shape[1]+i]
	}

	return B
}

func (A *Tensor) Set_Row(row int, B *Tensor) {

	// Check that the two Tensors are compatible
	if A.shape[1] != B.shape[1] {
		panic("Set_Row() --- Tensors must have the same number of columns")
	}

	// Copy the row into the new Tensor
	for i := 0; i < A.shape[1]; i++ {
		A.data[row*A.shape[1]+i] = B.data[i]
	}
}
