package TG

// elimination.go algorithms for solving systems of linear equations

import (
	"math"
)

// --------------------------------------------------------------------------------------------------Normal Gaussian Elimination

type GaussianEliminationOp struct{}

func (ge GaussianEliminationOp) Execute(tensors ...*Tensor) *Tensor {

	A, b := tensors[0], tensors[1]

	// Implement the Gaussian Elimination logic here.
	// You can use the existing Gaussian_Elimination function logic.
	if len(b.Shape) == 1 {
		b = b.Add_Singleton(0) // Augment_Matrix() requires a 2D Tensor
	}

	// Perform Forward Elimination and Back Substitution on the augmented matrix
	Ab := Augment_Matrix(A, b)
	Forward_Elimination(Ab)
	return Back_Substitution(Ab, A) // returns x
}

// Gaussian_Elimination() performs Gaussian Elimination on a system of linear equations. It takes 3 parameters. A, b, and batching.
// batching is a bool determining whether to use batched processing. A is an n x n matrix, b is an n x 1 column vector. It returns x of Ax=b
func Gaussian_Elimination(A *Tensor, b *Tensor, batching bool) *Tensor {

	GE := GaussianEliminationOp{} // create instance of GaussianElimination struct

	if batching {
		return BatchedOperation(GE, A, b) // batched processing
	} else {
		return GE.Execute(A, b) // single processing
	}
}

// --- ---------------------------------------------------------------------------------------------Gauss Jordan Elimination

type GaussJordanEliminationOp struct{}

func (gje GaussJordanEliminationOp) Execute(tensors ...*Tensor) *Tensor {

	A, b := tensors[0], tensors[1]

	if len(b.Shape) == 1 {
		b = b.Add_Singleton(0) // <--- Augment_Matrix() requires a 2D Tensor
	}

	// Perform Forward ELimination and Backsubstitiution on the augmented matrix
	Ab := Augment_Matrix(A, b)
	Forward_Elimination(Ab)
	RREF(Ab) // <--- convert to reduced row echelon form

	return Ab.Remove_Dim(1, 3) // <--- Remove all columns except the last one (whihc is the solution in the augmented matrix)
}

// This function performs Gauss Jordan Elimination on a system of linear equations. Which means that in addition to forward Propagation,
// The function also reduces the matrix to reduced row echelon form (RREF). It takes two parameters: A and b, where A is an n x n matrix
// and b is an n x 1 column vector. It returns a column vector x that is the solution to the system of linear equations.
func Gauss_Jordan_Elimination(A *Tensor, b *Tensor, batching bool) *Tensor {

	GJE := GaussJordanEliminationOp{} // create instance of GaussJordanElimination struct

	if batching {
		return BatchedOperation(GJE, A, b) // batched processing
	} else {
		return GaussJordanEliminationOp{}.Execute(A, b) // single processing
	}

}

//--------------------------------------------------------------------------------------------------Helper Functions for Elimination Functions

// This function reduces a matrix to reduced row echelon form (RREF). It takes one parameter: Ab, which is an n x (n + 1) augmented matrix.
// It returns nothing, but it modifies the matrix in place.
func RREF(Ab *Tensor) {

	// Convert to Reduced Row Echelon Form (RREF)
	for k := Ab.Shape[0] - 1; k >= 0; k-- { // <--- decrement through rows

		// Make the diagonal element 1
		f := Ab.Data[k*Ab.Shape[1]+k]
		if f == 0 {
			panic("RREF() --- No unique solution")
		}
		for j := k; j < Ab.Shape[0]+1; j++ {
			Ab.Data[k*Ab.Shape[1]+j] = Ab.Data[k*Ab.Shape[1]+j] / f // <--- Ab[k, j] = Ab[k, j] / f
		}

		// Make all elements above the current one 0
		for i := 0; i < k; i++ {
			f := Ab.Data[i*Ab.Shape[1]+k]
			for j := k; j < Ab.Shape[0]+1; j++ {
				Ab.Data[i*Ab.Shape[1]+j] = Ab.Data[i*Ab.Shape[1]+j] - Ab.Data[k*Ab.Shape[1]+j]*f // <--- Ab[i, j] = Ab[i, j] - Ab[k, j] * f
			}
		}
	}
}

// Move the forward elimination section of the above funciton into its own function
func Forward_Elimination(Ab *Tensor) {
	// Forward Elimination
	for k := 0; k < Ab.Shape[0]-1; k++ { // <---  iterate through first elements of each row

		// Find the k-th pivot
		i_max := Find_Pivot(Ab, k) // <--- find the row with the abs maximum element in column k
		if Ab.Data[i_max*Ab.Shape[1]+k] == 0 {
			panic("Forward_Elimination() within Gaussian_Elimination() --- No unique solution")
		}
		Ab.Swap_Rows(k, i_max) // <--- swap the k-th row with the row with the max element in column k

		// Make all rows below this one 0 in current column
		for i := k + 1; i < Ab.Shape[0]; i++ {
			f := Ab.Data[i*Ab.Shape[1]+k] / Ab.Data[k*Ab.Shape[1]+k] // <--- f = Ab[i, k] / Ab[k, k]

			// Subtract (f * k-th row) from i-th row
			for j := k; j < Ab.Shape[0]+1; j++ {
				Ab.Data[i*Ab.Shape[1]+j] = Ab.Data[i*Ab.Shape[1]+j] - Ab.Data[k*Ab.Shape[1]+j]*f
			}
		}
	}
}

// This function performs back substitution on an augmented matrix Ab and returns the solution to the system of linear
// equations. It takes two parameters: Ab and A, where Ab is an n x (n + 1) augmented matrix and A is an n x n matrix.
// It returns a column vector x that is the solution to the system of linear equations.
func Back_Substitution(Ab *Tensor, A *Tensor) *Tensor {
	// Back Substitution
	x := ZeroTensor([]int{Ab.Shape[0], 1}, false)
	for i := Ab.Shape[0] - 1; i >= 0; i-- { // <--- decrement through rows
		sum := 0.0
		for j := i + 1; j < Ab.Shape[0]; j++ {
			sum += Ab.Data[i*Ab.Shape[1]+j] * x.Data[j]
		}
		x.Data[i] = (Ab.Data[i*Ab.Shape[1]+A.Shape[0]] - sum) / Ab.Data[i*Ab.Shape[1]+i] // <--- x[i] = (Ab[i, n] - sum) / Ab[i, i]
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
	for i := k; i < A.Shape[0]; i++ {
		if math.Abs(A.Data[i*A.Shape[1]+k]) > max_val {
			i_max = i
			max_val = math.Abs(A.Data[i*A.Shape[1]+k])
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
	B := ZeroTensor([]int{1, A.Shape[1]}, false)

	// Copy the row into the new Tensor
	for i := 0; i < A.Shape[1]; i++ {
		B.Data[i] = A.Data[row*A.Shape[1]+i]
	}

	return B
}

// This is a helper function of Swap_Rows() above. It sets the specified row of A to the values of B.
func (A *Tensor) Set_Row(row int, B *Tensor) {

	// Check that the two Tensors are compatible
	if A.Shape[1] != B.Shape[1] {
		panic("Set_Row() --- Tensors must have the same number of columns")
	}

	// Copy the row into the new Tensor
	for i := 0; i < A.Shape[1]; i++ {
		A.Data[row*A.Shape[1]+i] = B.Data[i]
	}
}
