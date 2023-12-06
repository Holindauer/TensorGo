// NOTE: go over funcs/methods and be deliberate about why a func is a methods
// vs a func. Propbably make inplace methods methods and the rest funcs

// Functions I am intednding to implement are listed below

// unique elements of array

// argmax along a dimension

// argmin along a dimension

// covariance matrix computation

// normalization functions --- implement a few major strategies

// concatenate along an axis

// broadcasting

// At this stage in the project, main is not about directing any specific processes
// It is about testing the functionality of the code in the other files.
package main

import (
	"fmt"

	. "github.com/Holindauer/Tensor-Go.git/TG"
)

func main() {

	A := RandFloat_Tensor([]int{3, 3, 3}, 0, 1, true)

	fmt.Println("Original A:", A)

	Display_Matrix(A, true)

	fmt.Println("Sum_Axis(A, 0):")
	A_Sum_Axis := A.Sum_Axis(0, false)
	Display_Matrix(A_Sum_Axis, false)

	fmt.Println("Mean_Axis(A, 0):")
	A_Mean_Axis := A.Mean_Axis(0, false)
	Display_Matrix(A_Mean_Axis, false)

	fmt.Println("Var_Axis(A, 0):")
	A_Var_Axis := A.Var_Axis(0, false)
	Display_Matrix(A_Var_Axis, false)

	fmt.Println("Std_Axis(A, 0):")

	A_Std_Axis := A.Std_Axis(0, false)
	Display_Matrix(A_Std_Axis, false)

	//-------------------------------------------------------------------------------------------------------------- Check_Orthogonal(), Check_Acute(), Check_Obtuse()

	// fmt.Println("Testing Check_Orthogonal() unbatched...")
	// A := Zero_Tensor([]int{2}, false)
	// B := Zero_Tensor([]int{2}, false)

	// // orthogonal case
	// A.Data = []float64{2, -2}
	// B.Data = []float64{-2, 2}

	// fmt.Println("\n\n Orthogonal case: ")
	// fmt.Println("\nA = ", A)
	// fmt.Println("\nB = ", B)
	// fmt.Println("Expect: True --- Got: Check_Orthogonal(A, B, false) = ", Check_Orthogonal(A, B, false))
	// fmt.Println("Expect: False --- Got: Check_Acute(A, B, false) = ", Check_Acute(A, B, false))
	// fmt.Println("Expect: False --- Got: Check_Obtuse(A, B, false) = ", Check_Obtuse(A, B, false))

	// // acute case
	// A.Data = []float64{2, -2}
	// B.Data = []float64{2, 0}

	// fmt.Println("\n\n Acute case: ")
	// fmt.Println("\nA = ", A)
	// fmt.Println("\nB = ", B)
	// fmt.Println("Expect: False --- Got: Check_Orthogonal(A, B, false) = ", Check_Orthogonal(A, B, false))
	// fmt.Println("Expect: True --- Got: Check_Acute(A, B, false) = ", Check_Acute(A, B, false))
	// fmt.Println("Expect: False --- Got: Check_Obtuse(A, B, false) = ", Check_Obtuse(A, B, false))

	// // acute case
	// A.Data = []float64{-2, -2}
	// B.Data = []float64{2, 0}

	// fmt.Println("\n\n Obtuse case: ")
	// fmt.Println("\nA = ", A)
	// fmt.Println("\nB = ", B)
	// fmt.Println("Expect: False --- Got: Check_Orthogonal(A, B, false) = ", Check_Orthogonal(A, B, false))
	// fmt.Println("Expect: False --- Got: Check_Acute(A, B, false) = ", Check_Acute(A, B, false))
	// fmt.Println("Expect: True --- Got: Check_Obtuse(A, B, false) = ", Check_Obtuse(A, B, false))

}
