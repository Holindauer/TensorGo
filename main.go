package main

import (
	"fmt"
)

// At this stage in the project, main is not about directing any specific processes
// It is about testing the functionality of the code in the other files.
func main() {

	tensor := New_Tensor([]int{2, 3})

	fmt.Println("Tensor shape:", tensor.shape)
	fmt.Println("Tensor data:", tensor.data)

	index := getFlattenedIndex([]int{1, 2}, tensor.shape)
	fmt.Println("Flattened index:", index)

}
