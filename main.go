package main

import (
	"fmt"
)

// At this stage in the project, main is not about directing any specific processes
// It is about testing the functionality of the code in the other files.
func main() {

	// Run an example of all the functions in vector.go
	fmt.Println("Testing vector.go")
	t1 := Range_Tensor([]int{5})
	t2 := Range_Tensor([]int{5})
	fmt.Println("t1:", t1)
	fmt.Println("t2:", t2)
	fmt.Println("dot(t1, t2):", dot(t1, t2))
	fmt.Println("norm(t1):", Norm(t1))
	fmt.Println("unit(t1):", Unit(t1))
	fmt.Println("Check_Perpindicular(t1, t2):", Check_Perpendicular(t1, t2))
	fmt.Println("cosine_similar(t1, t2):", Cosine_Similarity(t1, t2))

	// Run an example of all the functions in matrix.go
	fmt.Println("Testing matrix.go")
	A := Range_Tensor([]int{3, 4})
	B := Range_Tensor([]int{4, 5})
	fmt.Println("A:", A)
	fmt.Println("B:", B)
	fmt.Println("matmul(A, B):", Matmul(A, B))

	// Run an example of all the functions in tensor.go
	fmt.Println("Testing tensor.go")
	t := Range_Tensor([]int{2, 3, 4})
	fmt.Println("t:", t)
	fmt.Println("t.shape:", t.shape)
	fmt.Println("t.data:", t.data)
	fmt.Println("Index([]int{1, 2, 3}, t.shape):", Index([]int{1, 2, 3}, t.shape))
	fmt.Println("Zero_Tensor([]int{2, 3, 4}):", Zero_Tensor([]int{2, 3, 4}))
	fmt.Println("Range_Tensor([]int{2, 3, 4}):", Range_Tensor([]int{2, 3, 4}))
	fmt.Println("Same_Shape(t, t):", Same_Shape(t, t))

	// Run an example of all the functions in general.go
	fmt.Println("Testing general.go")
	fmt.Println("Scalar_Mult_(t, 2):", Scalar_Mult_(t, 2))
	fmt.Println("Add(t, t):", Add(t, t))
	fmt.Println("Subtract(t, t):", Subtract(t, t))

}
