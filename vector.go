package main

// This source contains functions related to
// vector/single dimensional tensor operations

import (
	"math"
	"sync"
)

// this function checks if two Tensors are vectors of the same dimension
func Check_Vector_Compatibility(t1 *Tensor, t2 *Tensor) bool {

	// check if tensors are vectors
	if len(t1.shape) != 1 || len(t2.shape) != 1 {
		return false
	}

	// check if vectors are of same length
	if len(t1.data) != len(t2.data) {
		return false
	}

	return true
}

// This function computes the dot product of two vectors
func dot(t1 *Tensor, t2 *Tensor) float64 {

	// check if tensors are vectors
	if Check_Vector_Compatibility(t1, t2) == false {
		panic("Tensors must both be vectors to compute dot product")
	}

	numGoroutines := 4 // Adjust this value to control the number of goroutines
	chunkSize := len(t1.data) / numGoroutines
	results := make(chan float64, numGoroutines) // <-- Create a buffered channel to store the results

	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {

		wg.Add(1) // Increment the WaitGroup counter

		start := i * chunkSize //  compute bounds of the chunk
		end := start + chunkSize

		if i == numGoroutines-1 {
			end = len(t1.data) // Ensure the last chunk includes any remaining elements
		}
		go computeDot(t1, t2, start, end, results, &wg)
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
func computeDot(t1 *Tensor, t2 *Tensor, start int, end int, results chan<- float64, wg *sync.WaitGroup) {
	defer wg.Done() // Decrement the counter when the goroutine completes
	var sum float64
	for i := start; i < end; i++ {
		sum += t1.data[i] * t2.data[i] // <-- Compute the dot product for this chunk
	}
	results <- sum // <-- Write the result to the channel
}

// this function computes and returns the norm of a vector
func Norm(t *Tensor) float64 {

	// check if tensor is a vector
	if len(t.shape) != 1 {
		panic("Tensor must be a vector to compute norm")
	}

	return math.Sqrt(dot(t, t))
}

// this function returns the unit vector of a vector
// it checks if the norm is zero, and if so, returns the zero vector
func Unit(A *Tensor) *Tensor {

	// check if tensor is a vector
	if len(A.shape) != 1 {
		panic("Tensor must be a vector to compute unit vector")
	}

	norm := Norm(A) // compute norm of A

	if norm == 0 {
		return Zero_Tensor(A.shape)
	}

	// create a new tensor to store the unit vector
	B := Zero_Tensor(A.shape)

	// compute the unit vector
	for i := 0; i < len(A.data); i++ {
		B.data[i] = A.data[i] / norm
	}

	return B
}

// This function checks if two vectors are perpidicular
func Check_Perpendicular(t1 *Tensor, t2 *Tensor) bool {

	// check if tensors are vectors
	if Check_Vector_Compatibility(t1, t2) == false {
		panic("Tensors must both be vectors to check if perpendicular")
	}

	// check if the dot product is zero
	if dot(t1, t2) == 0 {
		return true
	}

	return false
}

// This function computes the cosine similarity of two vectors
func Cosine_Similarity(t1 *Tensor, t2 *Tensor) float64 {

	// check if tensors are vectors
	if Check_Vector_Compatibility(t1, t2) == false {
		panic("Tensors must both be vectors to compute cosine similarity")
	}

	return dot(t1, t2) / (Norm(t1) * Norm(t2))
}
