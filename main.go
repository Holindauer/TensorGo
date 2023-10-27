package main

import (
	"fmt"
)

// At this stage in the project, main is not about directing any specific processes
// It is about testing the functionality of the code in the other files.
func main() {

	vector_1 := New_Tensor([]int{3})
	vector_2 := New_Tensor([]int{3})

	vector_1.data = []float64{1, 2, 3}
	vector_2.data = []float64{4, 5, 6}

	// take dot product
	dot_product := dot(vector_1, vector_2)

	fmt.Println("Dot Product of vector 1 and 2: ", dot_product)

}
