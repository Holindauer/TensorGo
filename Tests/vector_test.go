package GLA

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func Test_Vector(t *testing.T) {

	// Test Dot() --- currently no batching
	fmt.Println("Testing Dot Product:\n---------------------")
	A := Range_Tensor([]int{4}, false)
	B := Range_Tensor([]int{4}, false)

	fmt.Println("\nA:", A.Data)
	fmt.Println("\nB:", B.Data)
	fmt.Println("\nDot Product of A and B:", Dot(A, B))

	// Test Norm() --- currently no batching
	fmt.Println("\nTesting Norm:\n-------------")
	A = Range_Tensor([]int{4}, false)
	fmt.Println("\nA:", A.Data)
	fmt.Println("\nNorm of A:", Norm(A))

	// Test Unit_Vector() --- currently no batching
	fmt.Println("\nTesting Unit Vector:\n--------------------")
	A = Range_Tensor([]int{4}, false)
	fmt.Println("\nA:", A.Data)
	fmt.Println("\nUnit Vector of A:", Unit(A).Data, "---- Norm of Unit(A):", Norm(Unit(A)))

	// Test Check_Perpindicular()
	fmt.Println("\nTesting Check_Perpindicular():\n------------------------------")
	P_1 := Range_Tensor([]int{2}, false)
	P_2 := Range_Tensor([]int{2}, false)

	P_1.Data = []float64{-1, 2}
	P_2.Data = []float64{4, 2}

	fmt.Println("\nA:", P_1.Data)
	fmt.Println("\nB:", P_2.Data)

	fmt.Println("\nExpected: true --- Check_Perpindicular(P_1, P_2)?:", Check_Perpendicular(P_1, P_2))

	P_1.Data = []float64{1, 2}
	P_2.Data = []float64{4, 2}

	fmt.Println("\nA:", P_1.Data)
	fmt.Println("\nB:", P_2.Data)

	fmt.Println("\nExpected: false --- Check_Perpindicular(P_1, P_2)?:", Check_Perpendicular(P_1, P_2))

	// Test Cosine_Similarity()
	fmt.Println("\nTesting Cosine_Similarity():\n------------------------------")

	A = Range_Tensor([]int{4}, false)
	B = Range_Tensor([]int{4}, false)

	fmt.Println("\nA:", A.Data)
	fmt.Println("\nA:", B.Data)

	fmt.Println("\nExpected: 1  --- Cosine_Similarity(P_1, P_2)?:", Cosine_Similarity(A, B))

	fmt.Println("\nP_1:", P_1.Data)
	fmt.Println("\nP_2:", P_2.Data)

	fmt.Println("\nExpected: 0.8 --- Cosine_Similarity(P_1, P_2)?:", Cosine_Similarity(P_1, P_2))

	// Test Outer_Product()
	fmt.Println("\nTesting Outer_Product():\n------------------------------")

	A = Range_Tensor([]int{4}, false)
	B = Range_Tensor([]int{4}, false)

	fmt.Println("\nA:", A.Data)
	fmt.Println("\nA:", B.Data)

	fmt.Println("\nExpected: [[0 0 0 0] [0 1 2 3] [0 2 4 6] [0 3 6 9]] --- Outer_Product(P_1, P_2)?:", Outer(A, B).Data)

}
