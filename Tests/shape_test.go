package GLA

// This source file contains tests of operations regarding shape manipulation operations frmo shape.go

import (
	"fmt"

	"testing"

	. "github.com/Holindauer/Go-LinAlg.git/GLA"
)

func Test_Shape(t *testing.T) {

	fmt.Println()
	fmt.Println()
	fmt.Println("Now Testing Shape Operations:\n===============================")

	// -------------------------------------------------------------------------Test Partial()
	fmt.Println("\n\nTesting Partial()... \n====================")

	A := Range_Tensor([]int{4, 4, 4, 4}, true)
	fmt.Println("A.Shape() = ", A.Shape)

	fmt.Println("Applying A.Partial(\":, 2:, 1:3, :3\")")
	A_Partial := A.Partial(":, 2:, 1:3, :3")

	fmt.Println("A_Partial.Shape() = ", A_Partial.Shape)
	fmt.Println()
	fmt.Println()

	A = Range_Tensor([]int{4, 3, 3}, true)
	fmt.Println("\nA.Shape() = ", A.Shape, "\n\nA:")
	Display_Matrix(A, true)

	fmt.Println("\nApplying A.Partial(\"0:1, :, :\")")
	A_Partial = A.Partial("0:1, :, :")
	fmt.Println("A_Partial.Shape() = ", A_Partial.Shape, "\n\nA_Partial:")
	Display_Matrix(A_Partial, true)

	// -------------------------------------------------------------------------Test Reshape()
	fmt.Println("\n\nTesting Reshape()...\n====================")

	A = Ones_Tensor([]int{2, 3, 4, 5}, true)
	fmt.Println("\nOnes Tensor A.Shape() = ", A.Shape, "\n\nA:")
	fmt.Println("Sum of All Elements fo A: ", A.Sum_All())

	fmt.Println("\nApplying A.Reshape([]int{3, 60})")
	A_Reshape := A.Reshape([]int{2, 60})
	fmt.Println("A_Reshape.Shape() = ", A_Reshape.Shape, "\n\nA_Reshape:")
	fmt.Println("Sum of All Elements of A_Reshape should be the same: ", A_Reshape.Sum_All())

	// -------------------------------------------------------------------------Test Transpose()

	fmt.Println("\n\nTesting Transpose()...\n====================")

	A = Range_Tensor([]int{5, 7}, false)

	fmt.Println("\nRange Tensor A.Shape() = ", A.Shape, "\n\nA:")
	Display_Matrix(A, false)

	A_Transpose := A.Transpose([]int{1, 0})
	fmt.Println("\nApplying A_Transpose := A.Transpose([]int{1, 0})")
	fmt.Println("A_Transpose.Shape() = ", A_Transpose.Shape, "\n\nA_Transpose:")
	Display_Matrix(A_Transpose, false)

	// higher dimmensional test
	A = Range_Tensor([]int{2, 3, 4, 5}, true)

	fmt.Println("\n\nRange Tensor A.Shape() = ", A.Shape)

	fmt.Println("\nApplying A_Transpose := A.Transpose([]int{0, 2, 1, 3})")
	A_Transpose = A.Transpose([]int{0, 2, 1, 3})
	fmt.Println("A_Transpose.Shape() = ", A_Transpose.Shape)

	fmt.Println("\nRetrieving from the the value at index [0, 1, 2, 3] in A: ")
	fmt.Println("A_Transpose.Get([]int{0, 1, 2, 3}) = ", A.Retrieve([]int{0, 1, 2, 3}))

	fmt.Println("\nRetrieving from the the value at index [0, 2, 1, 3] in A_Transpose: ")
	fmt.Println("A_Transpose.Get([]int{0, 2, 1, 3}) = ", A_Transpose.Retrieve([]int{0, 2, 1, 3}))

	// -------------------------------------------------------------------------Test Concat()

	fmt.Println("\n\nTesting Concat()...\n====================")

	A = Range_Tensor([]int{2, 3, 3}, true)
	B := RandFloat_Tensor([]int{2, 3, 3}, 0, 1, true)

	fmt.Println("\nRange Tensor A.Shape() = ", A.Shape, "\n\nA:")
	Display_Matrix(A, true)

	fmt.Println("\nRandFloat Tensor B.Shape() = ", B.Shape, "\n\nB:")
	Display_Matrix(B, true)

	fmt.Println("\nApplying A.Concat(B, 0)")
	A_Concat := A.Concat(B, 0)
	fmt.Println("A_Concat.Shape() = ", A_Concat.Shape, "\n\nA_Concat:")
	Display_Matrix(A_Concat, true)

	// -------------------------------------------------------------------------Test Extend_Shape()

	fmt.Println("\n\nTesting Extend_Shape()...\n====================")

	A = Range_Tensor([]int{3, 3}, false)

	fmt.Println("\nRange Tensor A.Shape() = ", A.Shape, "\n\nA:")
	Display_Matrix(A, true)

	fmt.Println("\nApplying A.Extend_Shape(2) should create a 3x3x2 Tensor with each 3x3 state in the dimmension of length  beinga copy of A")
	A_Extended := A.Extend_Shape(2)

	fmt.Println("A_Extended.Shape() = ", A_Extended.Shape, "\n\nA_Extended:")
	Display_Matrix(A_Extended, true)

	fmt.Println("Transposing to a 3x3x2 Tensor should correct the order of the 3x3 states")
	A_Extended_Transpose := A_Extended.Transpose([]int{2, 0, 1})
	fmt.Println("After applying A_Extended.Transpose([]int{2, 0, 1}), A_Extended_Transpose.Shape = ", A_Extended_Transpose.Shape, "\n\nA_Extended_Transpose:")
	Display_Matrix(A_Extended_Transpose, true)

	// -------------------------------------------------------------------------Test Extend_Dim()

	fmt.Println("\n\nTesting Extend_Dim()...\n====================")

	A = Ones_Tensor([]int{3, 3}, false)

	fmt.Println("\nOnes Tensor A.Shape() = ", A.Shape, "Sum of all elements of A: ", A.Sum_All(), "\n\nA:")
	Display_Matrix(A, false)

	fmt.Println("\nApplying A.Extend_Dim(0, 2) should extend the first dimmension of A by 2, initializing the new dimmension with 0s")
	A_Extended_Dim := A.Extend_Dim(0, 2)

	fmt.Println("A_Extended_Dim.Shape() = ", A_Extended_Dim.Shape, "Sum of all elements of A_Extended_Dim: ", A_Extended_Dim.Sum_All(), "\n\nA_Extended_Dim:")
	Display_Matrix(A_Extended_Dim, false)

	// -------------------------------------------------------------------------Test Add_Singleton()

	fmt.Println("\n\nTesting Add_Singleton()...\n====================")

	A = Range_Tensor([]int{3, 3}, false)

	fmt.Println("\nRange Tensor A.Shape() = ", A.Shape, "Sum of All Elements of A: ", A.Sum_All(), "\n\nA:")
	Display_Matrix(A, false)

	fmt.Println("\nApplying A.Add_Singleton(0, 2) should add a new dimmension of length 1 to the first dimmension of A")
	A_Add_Singleton := A.Add_Singleton()

	fmt.Println("A_Add_Singleton.Shape() = ", A_Add_Singleton.Shape, "Sum of All Elements of A_Add_Singleton: ", A_Add_Singleton.Sum_All(), "\n\nA_Add_Singleton:")

	// -------------------------------------------------------------------------Test Remove_Singletons()

	fmt.Println("\n\nTesting Remove_Singletons()...\n====================")

	A = Range_Tensor([]int{5, 5, 1, 1, 1, 1, 1, 1, 1}, false)

	fmt.Println("\nRange Tensor A.Shape() = ", A.Shape, "Sum of All Elements of A: ", A.Sum_All(), "\n\nA:")
	Display_Matrix(A, false)

	fmt.Println("\nApplying A.Remove_Singletons() should remove all singleton dimmensions from A")
	A_Remove_Singletons := A.Remove_Singletons()

	fmt.Println("A_Remove_Singletons.Shape() = ", A_Remove_Singletons.Shape, "Sum of All Elements of A_Remove_Singletons: ", A_Remove_Singletons.Sum_All(), "\n\nA_Remove_Singletons:")
	Display_Matrix(A_Remove_Singletons, false)

}
