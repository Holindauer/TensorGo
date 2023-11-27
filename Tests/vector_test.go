package GLA

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func Test_Vector(t *testing.T) {

	//-------------------------------------------------------------------------------------------------------------- Dot()

	fmt.Println("Testing Dot() unbatched...")
	A := Range_Tensor([]int{3}, false)
	B := Range_Tensor([]int{3}, false)
	C := Dot(A, B, false)
	fmt.Println("A = ", A)
	fmt.Println("B = ", B)
	fmt.Println("C = Dot(A, B, false) = ", C)
	fmt.Println()

	fmt.Println("Testing Dot() batched...")
	A = Range_Tensor([]int{3, 3}, true)
	B = Range_Tensor([]int{3, 3}, true)
	C = Dot(A, B, true)

	fmt.Println("A = ")
	Display_Matrix(A, true)
	fmt.Println("B = ")
	Display_Matrix(B, true)
	fmt.Println("C = Dot(A, B, true) = ", C)
	fmt.Println()

	//-------------------------------------------------------------------------------------------------------------- Norm()

	fmt.Println("Testing Norm() unbatched...")
	A = Range_Tensor([]int{3}, false)
	C = Norm(A, false)
	fmt.Println("A = ", A)
	fmt.Println("C = Norm(A, false) = ", C)
	fmt.Println()

	fmt.Println("Testing Norm() batched...")
	A = Range_Tensor([]int{3, 3}, true)
	C = Norm(A, true)
	Display_Matrix(A, true)
	fmt.Println("C = Norm(A, true) = ", C)
	fmt.Println()

	//-------------------------------------------------------------------------------------------------------------- Unit()

	fmt.Println("\nTesting Unit() unbatched...")
	A = Range_Tensor([]int{3}, false)
	C = Unit(A, false)
	fmt.Println("A = ", A)
	fmt.Println("C = Unit(A, false) = ", C)
	fmt.Println()

	fmt.Println("\nTesting Unit() batched...")
	A = Range_Tensor([]int{3, 3}, true)
	C = Unit(A, true)
	Display_Matrix(A, true)
	fmt.Println("\nC = Unit(A, true) = ", C)
	fmt.Println()

	//-------------------------------------------------------------------------------------------------------------- Check_Perpendicular()

	fmt.Println("Testing Check_Perpendicular() unbatched...")
	A = Zero_Tensor([]int{2}, false)
	B = Zero_Tensor([]int{2}, false)
	// case 1
	A.Data = []float64{-1, 2}
	B.Data = []float64{4, 2}

	fmt.Println("A = ", A)
	fmt.Println("B = ", B)
	fmt.Println("Check_Perpendicular(A, B, false) = ", Check_Perpendicular(A, B, false))

	// case 2
	A.Data = []float64{1, 2}
	B.Data = []float64{4, 2}

	fmt.Println("A = ", A)
	fmt.Println("B = ", B)
	fmt.Println("Check_Perpendicular(A, B, false) = ", Check_Perpendicular(A, B, false))

	// batched test
	fmt.Println("Testing Check_Perpendicular() batched...")
	A = Zero_Tensor([]int{3, 2}, true)
	B = Zero_Tensor([]int{3, 2}, true)

	// case 1
	A.Data = []float64{-1, 2, -1, 2, -1, 2}
	B.Data = []float64{4, 2, 4, 2, 4, 2}

	fmt.Println("\nA: ")
	Display_Matrix(A, true)
	fmt.Println("\nB: ")
	Display_Matrix(B, true)
	fmt.Println("Check_Perpendicular(A, B, true) = ", (Check_Perpendicular(A, B, true)))

	// case 2
	A.Data = []float64{1, 2, 1, 2, 1, 2}
	B.Data = []float64{4, 2, 4, 2, 4, 2}

	fmt.Println("\nA: ")
	Display_Matrix(A, true)
	fmt.Println("\nB: ")
	Display_Matrix(B, true)
	fmt.Println("Check_Perpendicular(A, B, true) = :", Check_Perpendicular(A, B, true))

	//-------------------------------------------------------------------------------------------------------------- Cosine_Similarity()

	fmt.Println("\nTesting Cosine_Similarity() unbatched...")

	// case 1
	A = Range_Tensor([]int{4}, false)
	B = Range_Tensor([]int{4}, false)

	fmt.Println("\nA:", A.Data)
	fmt.Println("\nA:", B.Data)

	fmt.Println("\nCosine_Similarity(P_1, P_2)?:", Cosine_Similarity(A, B, false))

	// case 2
	A = Ones_Tensor([]int{4}, false)
	fmt.Println("\nA:", A.Data)
	fmt.Println("\nB:", B.Data)

	fmt.Println("\nCosine_Similarity(P_1, P_2)?:", Cosine_Similarity(A, B, false))

	// batched test
	fmt.Println("\nTesting Cosine_Similarity() batched...")

	// case 1
	A = Range_Tensor([]int{3, 4}, true)
	B = Range_Tensor([]int{3, 4}, true)

	fmt.Println("\nA:", A.Data)
	fmt.Println("\nB:", B.Data)

	fmt.Println("\nCosine_Similarity(A, B)?:", Cosine_Similarity(A, B, true))

	// case 2
	A = Ones_Tensor([]int{3, 4}, true)
	fmt.Println("\nA:", A)
	fmt.Println("\nB:", B)

	fmt.Println("\nCosine_Similarity(A, B)?:", Cosine_Similarity(A, B, true))

	//-------------------------------------------------------------------------------------------------------------- Outer()

	fmt.Println("\nTesting Outer() unbatched...")
	A = Range_Tensor([]int{3}, false)
	B = Range_Tensor([]int{3}, false)

	fmt.Println("\nA:", A.Data)
	fmt.Println("\nB:", B.Data)

	fmt.Println("\nOuter(A, B)?:")
	Display_Matrix(Outer(A, B, false), false)

	// batched test
	fmt.Println("\nTesting Outer() batched...")
	A = Range_Tensor([]int{3, 3}, true)
	B = Range_Tensor([]int{3, 3}, true)

	fmt.Println("\nA:")
	Display_Matrix(A, true)
	fmt.Println("\nB:")
	Display_Matrix(B, true)

	fmt.Println("\nOuter(A, B:")
	Display_Matrix(Outer(A, B, true), true)

}
