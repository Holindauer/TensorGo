package TG

// TODO better test coverage need for vecotr functions

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

func Test_Dot(t *testing.T) {
	/// @notice Test Dot() Unbatched

	A := Zero_Tensor([]int{2}, false)
	A.Data = []float64{2, -2}

	B := Zero_Tensor([]int{2}, false)
	B.Data = []float64{2, 2}

	A_dot_B := Dot(A, B, false)

	if A_dot_B.Data[0] != 0 {
		t.Error("Dot() failed")
	}

	/// @notice Test Dot() Batched
	A = Zero_Tensor([]int{2, 2}, false)
	A.Data = []float64{2, -2, 2, -2}

	B = Zero_Tensor([]int{2, 2}, false)
	B.Data = []float64{2, 2, 2, 2}

	A_dot_B = Dot(A, B, true)

	if A_dot_B.Data[0] != 0 || A_dot_B.Data[1] != 0 {
		t.Error("Dot() failed")
	}
}

func Test_Norm(t *testing.T) {
	/// @notice Test Norm() Unbatched

	A := Zero_Tensor([]int{2}, false)
	A.Data = []float64{4, 4}

	A_norm := A.Norm(false)

	if A_norm.Data[0] != 5.656854249492381 {
		t.Error("Norm() failed")
	}

	/// @notice Test Norm() Batched
	A = Zero_Tensor([]int{2, 2}, false)
	A.Data = []float64{4, 4, 4, 4}

	A_norm = A.Norm(true)

	if A_norm.Data[0] != 5.656854249492381 || A_norm.Data[1] != 5.656854249492381 {
		t.Error("Norm() failed")
	}
}

func Test_Unit(t *testing.T) {
	/// @notice Test Unit() Unbatched

	A := Zero_Tensor([]int{2}, false)
	A.Data = []float64{4, 4}

	A_unit := A.Unit(false)

	if A_unit.Data[0] != 0.7071067811865475 || A_unit.Data[1] != 0.7071067811865475 {
		t.Error("Unit() failed")
	}

	/// @notice Test Unit() Batched
	A = Zero_Tensor([]int{2, 2}, false)
	A.Data = []float64{4, 4, 4, 4}

	A_unit = A.Unit(true)

	if A_unit.Data[0] != 0.7071067811865475 || A_unit.Data[1] != 0.7071067811865475 || A_unit.Data[2] != 0.7071067811865475 || A_unit.Data[3] != 0.7071067811865475 {
		t.Error("Unit() failed")
	}
}
