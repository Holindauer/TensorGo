package main

// This file contains algorithms for solving systems of linear equations

// Gaussian Elimination

// Plan for implementing Gaussian Elimination:

// I will need to implement a concat() function that concatenates two tensors along a specified axis.
// This will be used to build out the upper triangular matrix while keeping the original matrix intact.
// I will also need to implement a swap() function that swaps two rows of a matrix for cases where the
// pivot is zero. This will be used to swap rows to avoid dividing by zero.
// I'll also need to figure out some kind of simple equation solver for back substitution.

// --------------------------------------------------------------------------------------------------Normal Gaussian Elimination

/*
Algorithm GaussianElimination(A, b)
    Input: A is an n x n matrix, b is an n x 1 column vector
    Output: x is the solution to the linear system Ax = b

    // Augment matrix A with column vector b to form an n x (n+1) matrix Ab
    Ab = augment(A, b)

    // Forward Elimination
    for k from 1 to n-1
        // Find the k-th pivot
        i_max = findPivot(Ab, k)
        if Ab[i_max, k] == 0
            throw NoUniqueSolutionException
        swapRows(Ab, k, i_max)

        // Make all rows below this one 0 in current column
        for i from k+1 to n
            f = Ab[i, k] / Ab[k, k]
            // Subtract (f * k-th row) from i-th row
            for j from k to n+1
                Ab[i, j] = Ab[i, j] - Ab[k, j] * f

    // Back Substitution
    x = new array of size n
    for i from n down to 1
        sum = 0
        for j from i+1 to n
            sum = sum + Ab[i, j] * x[j]
        x[i] = (Ab[i, n+1] - sum) / Ab[i, i]

    return x

Procedure augment(A, b)
    // Create a new matrix with an extra column
    n = number of rows in A
    Ab = new matrix of size n x (n+1)
    for i from 1 to n
        for j from 1 to n
            Ab[i, j] = A[i, j]
        Ab[i, n+1] = b[i]
    return Ab

Function findPivot(Ab, k)
    // Find the row with the maximum element in column k
    i_max = k
    max_val = 0
    for i from k to n
        if abs(Ab[i, k]) > max_val
            i_max = i
            max_val = abs(Ab[i, k])
    return i_max

Procedure swapRows(Ab, i, j)
    // Swap the i-th and j-th rows of Ab
    temp = Ab[i]
    Ab[i] = Ab[j]
    Ab[j] = temp

*/

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
