package GLA

// This source contains functions related to
// vector/single dimensional tensor operations

import (
	"math"
	"sync"
)

// this function checks if two Tensors are vectors of the same dimension
func Check_Vector_Compatibility(t1 *Tensor, t2 *Tensor) bool {

	// check if tensors are vectors
	if len(t1.Shape) != 1 || len(t2.Shape) != 1 {
		return false
	}

	// check if vectors are of same length
	if len(t1.Data) != len(t2.Data) {
		return false
	}

	return true
}

//---------------------------------------------------------------------------------------------------------------------------- dot()

// This function computes the dot product of two vectors
func Dot(A *Tensor, B *Tensor) float64 {

	// check if tensors are vectors
	if Check_Vector_Compatibility(A, B) == false {
		panic("Within dot(): Tensors must both be vectors to compute dot product")
	}

	numGoroutines := 4 // Adjust this value to control the number of goroutines
	chunkSize := len(A.Data) / numGoroutines
	results := make(chan float64, numGoroutines) // <-- Create a buffered channel to store the results

	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {

		wg.Add(1) // Increment the WaitGroup counter

		start := i * chunkSize //  compute bounds of the chunk
		end := start + chunkSize

		if i == numGoroutines-1 {
			end = len(A.Data) // Ensure the last chunk includes any remaining elements
		}
		go computeDot(A, B, start, end, results, &wg)
	}

	wg.Wait() // Wait for all goroutines to finish
	close(results)

	var dot float64
	for result := range results {
		dot += result // <-- Accumulate the partial results
	}

	return dot
}

// This is a helper function for dot() above. It computes the dot product of a chunk of the vectors
func computeDot(A *Tensor, B *Tensor, start int, end int, results chan<- float64, wg *sync.WaitGroup) {
	defer wg.Done() // Decrement the counter when the goroutine completes
	var sum float64
	for i := start; i < end; i++ {
		sum += A.Data[i] * B.Data[i] // <-- Compute the dot product for this chunk
	}
	results <- sum // <-- Write the result to the channel
}

//---------------------------------------------------------------------------------------------------------------------------- Norm()

// this function computes and returns the norm of a vector
func Norm(A *Tensor) float64 {

	// check if tensor is a vector
	if len(A.Shape) != 1 {
		panic("Within Norm(): Tensor must be a vector to compute norm")
	}

	return math.Sqrt(Dot(A, A))
}

// this function returns the unit vector of a vector
// it checks if the norm is zero, and if so, returns the zero vector
func Unit(A *Tensor) *Tensor {

	// check if tensor is a vector
	if len(A.Shape) != 1 {
		panic("Within Unit(): Tensor must be a vector to compute unit vector")
	}

	norm := Norm(A) // compute norm of A

	if norm == 0 {
		return Zero_Tensor(A.Shape, false)
	}

	// create a new tensor to store the unit vector
	B := Zero_Tensor(A.Shape, false)

	// compute the unit vector
	for i := 0; i < len(A.Data); i++ {
		B.Data[i] = A.Data[i] / norm
	}

	return B
}

// This function checks if two vectors are perpidicular
func Check_Perpendicular(A *Tensor, B *Tensor) bool {

	// check if tensors are vectors
	if Check_Vector_Compatibility(A, B) == false {
		panic("Within Check_Perpindicular(): Tensors must both be vectors to check if perpendicular")
	}

	// check if the dot product is zero
	if Dot(A, B) == 0 {
		return true
	}

	return false
}

// This function computes the cosine similarity of two vectors
func Cosine_Similarity(A *Tensor, B *Tensor) float64 {

	// check if tensors are vectors
	if Check_Vector_Compatibility(A, B) == false {
		panic("Wihtin Cosine_Similarity(): Tensors must both be vectors to compute cosine similarity")
	}

	return Dot(A, B) / (Norm(A) * Norm(B))
}

// This function computes the outer product of two vectors
// it returns a pointer to a new 2D tensor
func Outer(A *Tensor, B *Tensor) *Tensor {

	// check if tensors are vectors
	if !(len(A.Shape) == 1 && len(B.Shape) == 1) {
		panic("Within Outer_Product(): Tensors must both be vectors to compute outer product")
	}

	// create a new tensor to store the result
	C := Zero_Tensor([]int{len(A.Data), len(B.Data)}, false)

	// compute the outer product
	for i := 0; i < len(A.Data); i++ {
		for j := 0; j < len(B.Data); j++ {
			C.Data[i*len(B.Data)+j] = A.Data[i] * B.Data[j] // Cij = ai * bj
		}
	}

	return C
}
