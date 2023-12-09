package TG

import (
	"fmt"
	"testing"

	//"math"
	. "github.com/Holindauer/Tensor-Go.git/TG"
)

func Test_Vector(t *testing.T) {

	fmt.Println("\n\nNow Testing Functions from vector.go\n-------------------------------------")

	// Test Dot()
	fmt.Print("Testing Dot() Unbatched...")
	A := Zero_Tensor([]int{2}, false)
	A.Data = []float64{2, -2}

	B := Zero_Tensor([]int{2}, false)
	B.Data = []float64{2, 2}

	A_dot_B := Dot(A, B, false)

	if A_dot_B.Data[0] != 0 {
		t.Error("Dot() failed")
	}

	fmt.Println("Success!")

	fmt.Print("Testing Dot() Batched...")
	A = Zero_Tensor([]int{2, 2}, false)
	A.Data = []float64{2, -2, 2, -2}

	B = Zero_Tensor([]int{2, 2}, false)
	B.Data = []float64{2, 2, 2, 2}

	A_dot_B = Dot(A, B, true)

	if A_dot_B.Data[0] != 0 || A_dot_B.Data[1] != 0 {
		t.Error("Dot() failed")
	}

	fmt.Println("Success!")

	// Test Norm()

	fmt.Print("Testing Norm() Unbatched...")
	A = Zero_Tensor([]int{2}, false)
	A.Data = []float64{4, 4}

	A_norm := A.Norm(false)

	if A_norm.Data[0] != 5.656854249492381 {
		t.Error("Norm() failed")
	}
	fmt.Println("Success!")

	fmt.Print("Testing Norm() Batched...")
	A = Zero_Tensor([]int{2, 2}, true)
	A.Data = []float64{4, 4, 4, 4}

	A_norm = A.Norm(true)

	// Test Unit()

	fmt.Print("Testing Unit() Unbatched...")
	A = Zero_Tensor([]int{2}, false)
	A.Data = []float64{4, 4}

	A_unit := A.Unit(false)

	if A_unit.Data[0] != 0.7071067811865475 || A_unit.Data[1] != 0.7071067811865475 {
		t.Error("Unit() failed")
	}

	fmt.Println("Success!")

	fmt.Print("Testing Unit() Batched...")
	A = Zero_Tensor([]int{2, 2}, true)
	A.Data = []float64{4, 4, 4, 4}

	A_unit = A.Unit(true)

	if A_unit.Data[0] != 0.7071067811865475 || A_unit.Data[1] != 0.7071067811865475 || A_unit.Data[2] != 0.7071067811865475 || A_unit.Data[3] != 0.7071067811865475 {
		t.Error("Unit() failed")
	}
	fmt.Println("Success!")

	// // Check Check_Orthogonal()

	// fmt.Println("Testing Check_Orthogonal(), Check_Acute(), and Check_Orthoganal unbatched...")
	// A = Zero_Tensor([]int{2}, false)
	// B = Zero_Tensor([]int{2}, false)

	// // orthogonal case
	// A.Data = []float64{2, -2}
	// B.Data = []float64{-2, -2}

	// if Check_Orthogonal(A, B, false).BoolData[0] != true {
	// 	t.Error("Check_Orthogonal() failed")
	// }
	// if Check_Acute(A, B, false).BoolData[0] != false {
	// 	t.Error("Check_Acute() failed")
	// }
	// if Check_Obtuse(A, B, false).BoolData[0] != false {
	// 	t.Error("Check_Obtuse() failed")
	// }

	// // acute case
	// A.Data = []float64{2, 1}
	// B.Data = []float64{2, 0}

	// if Check_Orthogonal(A, B, false).BoolData[0] != false {
	// 	t.Error("Check_Orthogonal() failed")
	// }
	// if Check_Acute(A, B, false).BoolData[0] != true {
	// 	t.Error("Check_Acute() failed")
	// }
	// if Check_Obtuse(A, B, false).BoolData[0] != false {
	// 	t.Error("Check_Obtuse() failed")
	// }

	// // obtuse case
	// A.Data = []float64{-2, -2}
	// B.Data = []float64{2, 0}

	// if Check_Orthogonal(A, B, false).BoolData[0] != false {
	// 	t.Error("Check_Orthogonal() failed")
	// }
	// if Check_Acute(A, B, false).BoolData[0] != false {
	// 	t.Error("Check_Acute() failed")
	// }
	// if Check_Obtuse(A, B, false).BoolData[0] != true {
	// 	t.Error("Check_Obtuse() failed")
	// }

	// fmt.Println("Success!")

	// fmt.Println("Testing Check_Orthogonal(), Check_Acute(), and Check_Orthoganal batched...")

	// A = Zero_Tensor([]int{2, 2}, true)
	// B = Zero_Tensor([]int{2, 2}, true)

	// // orthogonal case
	// A.Data = []float64{2, -2, 2, -2}
	// B.Data = []float64{-2, -2, -2, -2}

	// if Check_Orthogonal(A, B, true).BoolData[0] != true || Check_Orthogonal(A, B, true).BoolData[1] != true {
	// 	t.Error("Check_Orthogonal() failed")
	// }
	// if Check_Acute(A, B, true).BoolData[0] != false || Check_Acute(A, B, true).BoolData[1] != false {
	// 	t.Error("Check_Acute() failed")
	// }
	// if Check_Obtuse(A, B, true).BoolData[0] != false || Check_Obtuse(A, B, true).BoolData[1] != false {
	// 	t.Error("Check_Obtuse() failed")
	// }

	// // acute case
	// A.Data = []float64{2, 1, 2, 1}
	// B.Data = []float64{2, 0, 2, 0}

	// if Check_Orthogonal(A, B, true).BoolData[0] != false || Check_Orthogonal(A, B, true).BoolData[1] != false {
	// 	t.Error("Check_Orthogonal() failed")
	// }
	// if Check_Acute(A, B, true).BoolData[0] != true || Check_Acute(A, B, true).BoolData[1] != true {
	// 	t.Error("Check_Acute() failed")
	// }
	// if Check_Obtuse(A, B, true).BoolData[0] != false || Check_Obtuse(A, B, true).BoolData[1] != false {
	// 	t.Error("Check_Obtuse() failed")
	// }

	// // obtuse case
	// A.Data = []float64{-2, -2, -2, -2}
	// B.Data = []float64{2, 0, 2, 0}

	// if Check_Orthogonal(A, B, true).BoolData[0] != false || Check_Orthogonal(A, B, true).BoolData[1] != false {
	// 	t.Error("Check_Orthogonal() failed")
	// }
	// if Check_Acute(A, B, true).BoolData[0] != false || Check_Acute(A, B, true).BoolData[1] != false {
	// 	t.Error("Check_Acute() failed")
	// }
	// if Check_Obtuse(A, B, true).BoolData[0] != true || Check_Obtuse(A, B, true).BoolData[1] != true {
	// 	t.Error("Check_Obtuse() failed")
	// }

}
