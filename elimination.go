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

	if len(b.shape) == 1 {
		b = b.Add_Singleton() // <--- Augment_Matrix() requires a 2D Tensor
	}

	// Perform Forward ELimination and Backsubstitiution on the augmented matrix
	Ab := Augment_Matrix(A, b)
	Forward_Elimination(Ab)
	return Back_Substitution(Ab, A) // <--- returns x
}

// This function performs Gauss Jordan Elimination on a system of linear equations. Which means that in addition to forward Propagation,
// The function also reduces the matrix to reduced row echelon form (RREF). It takes two parameters: A and b, where A is an n x n matrix
// and b is an n x 1 column vector. It returns a column vector x that is the solution to the system of linear equations.
func Gauss_Jordan_Elimination(A *Tensor, b *Tensor) *Tensor {

	if len(b.shape) == 1 {
		b = b.Add_Singleton() // <--- Augment_Matrix() requires a 2D Tensor
	}

	// Perform Forward ELimination and Backsubstitiution on the augmented matrix
	Ab := Augment_Matrix(A, b)
	Forward_Elimination(Ab)
	RREF(Ab) // <--- convert to reduced row echelon form

	return Ab.Remove_Dim(1, 3) // <--- Remove all columns except the last one (whihc is the solution in the augmented matrix)
}

func RREF(Ab *Tensor) {

	// Convert to Reduced Row Echelon Form (RREF)
	for k := Ab.shape[0] - 1; k >= 0; k-- { // <--- decrement through rows

		// Make the diagonal element 1
		f := Ab.data[k*Ab.shape[1]+k]
		if f == 0 {
			panic("RREF() --- No unique solution")
		}
		for j := k; j < Ab.shape[0]+1; j++ {
			Ab.data[k*Ab.shape[1]+j] = Ab.data[k*Ab.shape[1]+j] / f // <--- Ab[k, j] = Ab[k, j] / f
		}

		// Make all elements above the current one 0
		for i := 0; i < k; i++ {
			f := Ab.data[i*Ab.shape[1]+k]
			for j := k; j < Ab.shape[0]+1; j++ {
				Ab.data[i*Ab.shape[1]+j] = Ab.data[i*Ab.shape[1]+j] - Ab.data[k*Ab.shape[1]+j]*f // <--- Ab[i, j] = Ab[i, j] - Ab[k, j] * f
			}
		}
	}
}

// Move the forward elimination section of the above funciton into its own function
func Forward_Elimination(Ab *Tensor) {
	// Forward Elimination
	for k := 0; k < Ab.shape[0]-1; k++ { // <---  iterate through first elements of each row

		// Find the k-th pivot
		i_max := Find_Pivot(Ab, k) // <--- find the row with the abs maximum element in column k
		if Ab.data[i_max*Ab.shape[1]+k] == 0 {
			panic("Forward_Elimination() within Gaussian_Elimination() --- No unique solution")
		}
		Ab.Swap_Rows(k, i_max) // <--- swap the k-th row with the row with the max element in column k

		// Make all rows below this one 0 in current column
		for i := k + 1; i < Ab.shape[0]; i++ {
			f := Ab.data[i*Ab.shape[1]+k] / Ab.data[k*Ab.shape[1]+k] // <--- f = Ab[i, k] / Ab[k, k]

			// Subtract (f * k-th row) from i-th row
			for j := k; j < Ab.shape[0]+1; j++ {
				Ab.data[i*Ab.shape[1]+j] = Ab.data[i*Ab.shape[1]+j] - Ab.data[k*Ab.shape[1]+j]*f
			}
		}
	}
}

// This function performs back substitution on an augmented matrix Ab and returns the solution to the system of linear
// equations. It takes two parameters: Ab and A, where Ab is an n x (n + 1) augmented matrix and A is an n x n matrix.
// It returns a column vector x that is the solution to the system of linear equations.
func Back_Substitution(Ab *Tensor, A *Tensor) *Tensor {
	// Back Substitution
	x := Zero_Tensor([]int{Ab.shape[0], 1})
	for i := Ab.shape[0] - 1; i >= 0; i-- { // <--- decrement through rows
		sum := 0.0
		for j := i + 1; j < Ab.shape[0]; j++ {
			sum += Ab.data[i*Ab.shape[1]+j] * x.data[j]
		}
		x.data[i] = (Ab.data[i*Ab.shape[1]+A.shape[0]] - sum) / Ab.data[i*Ab.shape[1]+i] // <--- x[i] = (Ab[i, n] - sum) / Ab[i, i]
	}
	return x
}

// This function finds the pivot of a matrix A in column k for use inside the call of Forward_Propagation() within the
// Gaussian_Elimination() function. This just means that it it finds the max row in column k and returns the index of that
// row. Beck in the Forward_Elimination() function, this row is swapped with the k-th row.
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

// This is a helper function of Gaussian_Elimination() above. It swaps two rows of A.
func (A *Tensor) Swap_Rows(i int, j int) {

	// Swap the i-th and j-th rows of A
	temp := A.Get_Row(i).Copy()
	A.Set_Row(i, A.Get_Row(j))
	A.Set_Row(j, temp)
}

// This is a helper function for Swap_Rows() above. It returns a copy of the specified row of A.
func (A *Tensor) Get_Row(row int) *Tensor {

	// Create a new Tensor to store the row
	B := Zero_Tensor([]int{1, A.shape[1]})

	// Copy the row into the new Tensor
	for i := 0; i < A.shape[1]; i++ {
		B.data[i] = A.data[row*A.shape[1]+i]
	}

	return B
}

// This is a helper function of Swap_Rows() above. It sets the specified row of A to the values of B.
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
