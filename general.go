package main

// This file contains functions for Tensor Operations, irregardless of dimensionality

// This funciton performs scalar multiplication on a tensor in place
// It returns a pointer to the same tensor
func Scalar_Mult_(A *Tensor, scalar float64) *Tensor {

	for i := 0; i < len(A.data); i++ {
		A.data[i] *= scalar
	}

	return A // <-- Pointer to the same tensor
}

// This function performs elementwise addition on two tensors
// The tensors must have the same shape. It returns a pointer to a new tensor
func Add(A *Tensor, B *Tensor) *Tensor {

	// Check that the tensors have the same shape
	if !Same_Shape(A, B) {
		panic("Tensors must have the same shape")
	}

	// Create a new tensor to hold the result
	C := Zero_Tensor(A.shape)

	// Perform the elementwise addition
	for i := 0; i < len(A.data); i++ {
		C.data[i] = A.data[i] + B.data[i]
	}

	return C // <-- Pointer to the new tensor
}

// This function performs elementwise subtraction on two tensors
// The tensors must have the same shape. It returns a pointer to a new tensor
func Subtract(A *Tensor, B *Tensor) *Tensor {

	// Check that the tensors have the same shape
	if !Same_Shape(A, B) {
		panic("Tensors must have the same shape")
	}

	// Create a new tensor to hold the result
	C := Zero_Tensor(A.shape)

	// Perform the elementwise subtraction
	for i := 0; i < len(A.data); i++ {
		C.data[i] = A.data[i] - B.data[i]
	}

	return C // <-- Pointer to the new tensor
}
