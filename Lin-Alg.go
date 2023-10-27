package main

import (
	"sync"
)

type Tensor struct {
	shape []int
	data  []float64 // <--- this 1D slice to store flattened tensor
}

// This function is used to create a new tensor It takes
// in a shape and returns a Tensor pointer with that shape
func New_Tensor(shape []int) *Tensor {

	t := new(Tensor) //  <--- this is a pointer to a tensor
	t.shape = shape
	t.data = make([]float64, len(shape)) // create slice of floats for data

	return t
}

// The general algorithm for computing the index of a flattened tensor from the multi dimensional indices:
// Create a slice of ints to store the strides. A stride is the number of elements in the tensor that must
// be skipped to move one index in a given dimension. Then, iterate over through each dimension of the tensor,
// multiplying the stride of that dimmension by the index of that dimension. Add the result to the flattened index.
func getFlattenedIndex(indices []int, dims []int) int {

	strides := make([]int, len(dims)) // create a slice of ints to store the strides
	strides[len(dims)-1] = 1          // stride for the last dimension is always 1

	for i := len(dims) - 2; i >= 0; i-- { // iterate backwards through the dimensions
		strides[i] = strides[i+1] * dims[i+1] // multiply the stride of the current dimension by the size of the next dimension
		// this is because if you move one element up in dim i, then you must skip the entire
		// next dimension of the flattened tensor to get there
	}

	flattenedIndex := 0

	// iterate through tensor indices
	for i, index := range indices {
		// multiply the index by the stride of that dimension
		flattenedIndex += index * strides[i]
	}

	return flattenedIndex
}

func dot(t1 *Tensor, t2 *Tensor) float64 {

	// check if tensors are vectors
	if len(t1.shape) != len(t2.shape) {
		panic("Tensors must both be vectors to compute dot product")
	}

	// check if vectors are of same length
	if len(t1.data) != len(t2.data) {
		panic("Vectors must be of the same length to compute dot product")
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

func computeDot(t1 *Tensor, t2 *Tensor, start int, end int, results chan<- float64, wg *sync.WaitGroup) {
	defer wg.Done() // Decrement the counter when the goroutine completes
	var sum float64
	for i := start; i < end; i++ {
		sum += t1.data[i] * t2.data[i] // <-- Compute the dot product for this chunk
	}
	results <- sum // <-- Write the result to the channel
}
