package TG

// /*
// * @notice This file contains tests for functions in elimination.go
//  */

// import (
// 	"testing"

// 	. "github.com/Holindauer/Tensor-Go/TG"
// )

// func Test_Gaussian_Elimination(t *testing.T) {

// 	/// @notice Testing Gaussian_Elimination() unbatched

// 	// creat an A and x, multiply them to get b --- the solution will be x = [-1, 2, 2]
// 	A := Zero_Tensor([]int{3, 3}, false)

// 	A.Data[0].Scalar = 2
// 	A.Data[1].Scalar = 4
// 	A.Data[2].Scalar = -2
// 	A.Data[3].Scalar = 4
// 	A.Data[4].Scalar = 9
// 	A.Data[5].Scalar = -3
// 	A.Data[6].Scalar = -2
// 	A.Data[7].Scalar = -3
// 	A.Data[8].Scalar = 7

// 	x := Zero_Tensor([]int{3, 1}, false)

// 	x.Data[0].Scalar = -1
// 	x.Data[1].Scalar = 2
// 	x.Data[2].Scalar = 2

// 	b := MatMul(A, x, false)

// 	// solve for x
// 	x_solved := Gaussian_Elimination(A, b, false)

// 	// check that the solution is correct (within 1e-10)
// 	if x_solved.Data[0].Scalar < -1.0000000001 || x_solved.Data[0].Scalar > -0.9999999999 {
// 		t.Error("Gaussian_Elimination() failed")
// 	}
// 	if x_solved.Data[1].Scalar < 1.9999999999 || x_solved.Data[1].Scalar > 2.0000000001 {
// 		t.Error("Gaussian_Elimination() failed")
// 	}
// 	if x_solved.Data[2].Scalar < 1.9999999999 || x_solved.Data[2].Scalar > 2.0000000001 {
// 		t.Error("Gaussian_Elimination() failed")
// 	}

// 	/// @notice Testing Gaussian_Elimination() batched

// 	A = Zero_Tensor([]int{3, 3, 3}, true)

// 	A.Data[0].Scalar = 2
// 	A.Data[1].Scalar = 4
// 	A.Data[2].Scalar = -2
// 	A.Data[3].Scalar = 4
// 	A.Data[4].Scalar = 9
// 	A.Data[5].Scalar = -3
// 	A.Data[6].Scalar = -2
// 	A.Data[7].Scalar = -3
// 	A.Data[8].Scalar = 7
// 	A.Data[9].Scalar = 2
// 	A.Data[10].Scalar = 4
// 	A.Data[11].Scalar = -2
// 	A.Data[12].Scalar = 4
// 	A.Data[13].Scalar = 9
// 	A.Data[14].Scalar = -3
// 	A.Data[15].Scalar = -2
// 	A.Data[16].Scalar = -3
// 	A.Data[17].Scalar = 7
// 	A.Data[18].Scalar = 2
// 	A.Data[19].Scalar = 4
// 	A.Data[20].Scalar = -2
// 	A.Data[21].Scalar = 4
// 	A.Data[22].Scalar = 9
// 	A.Data[23].Scalar = -3
// 	A.Data[24].Scalar = -2
// 	A.Data[25].Scalar = -3
// 	A.Data[26].Scalar = 7

// 	x = Zero_Tensor([]int{3, 3, 1}, true)

// 	x.Data[0].Scalar = -1
// 	x.Data[1].Scalar = 2
// 	x.Data[2].Scalar = 2
// 	x.Data[3].Scalar = -1
// 	x.Data[4].Scalar = 2
// 	x.Data[5].Scalar = 2
// 	x.Data[6].Scalar = -1
// 	x.Data[7].Scalar = 2
// 	x.Data[8].Scalar = 2

// 	b = MatMul(A, x, true)

// 	// solve for x
// 	x_solved = Gaussian_Elimination(A, b, true)

// 	// check that the solution is correct (within 1e-10)
// 	if x_solved.Data[0].Scalar+x_solved.Data[3].Scalar+x_solved.Data[6].Scalar < -3.0000000001 || x_solved.Data[0].Scalar+x_solved.Data[3].Scalar+x_solved.Data[6].Scalar > -2.9999999999 {
// 		t.Error("Gaussian_Elimination() failed")
// 	}
// 	if x_solved.Data[1].Scalar+x_solved.Data[4].Scalar+x_solved.Data[7].Scalar < 5.9999999999 || x_solved.Data[1].Scalar+x_solved.Data[4].Scalar+x_solved.Data[7].Scalar > 6.0000000001 {
// 		t.Error("Gaussian_Elimination() failed")
// 	}
// 	if x_solved.Data[2].Scalar+x_solved.Data[5].Scalar+x_solved.Data[8].Scalar < 5.9999999999 || x_solved.Data[2].Scalar+x_solved.Data[5].Scalar+x_solved.Data[8].Scalar > 6.0000000001 {
// 		t.Error("Gaussian_Elimination() failed")
// 	}
// }

// func Test_Gauss_Jordan_Elimination(t *testing.T) {

// 	/// @notice Testing Gauss_Jordan_Elimination() unbatched

// 	// creat an A and x, multiply them to get b
// 	A := Zero_Tensor([]int{3, 3}, false)

// 	A.Data[0].Scalar = 2
// 	A.Data[1].Scalar = 4
// 	A.Data[2].Scalar = -2
// 	A.Data[3].Scalar = 4
// 	A.Data[4].Scalar = 9
// 	A.Data[5].Scalar = -3
// 	A.Data[6].Scalar = -2
// 	A.Data[7].Scalar = -3
// 	A.Data[8].Scalar = 7

// 	x := Zero_Tensor([]int{3, 1}, false)

// 	x.Data[0].Scalar = -1
// 	x.Data[1].Scalar = 2
// 	x.Data[2].Scalar = 2

// 	b := MatMul(A, x, false)

// 	// solve for x
// 	x_solved := Gauss_Jordan_Elimination(A, b, false)

// 	// check that the solution is correct (within 1e-10)
// 	if x_solved.Data[0].Scalar < -1.0000000001 || x_solved.Data[0].Scalar > -0.9999999999 {
// 		t.Error("Gauss_Jordan_Elimination() failed")
// 	}
// 	if x_solved.Data[1].Scalar < 1.9999999999 || x_solved.Data[1].Scalar > 2.0000000001 {
// 		t.Error("Gauss_Jordan_Elimination() failed")
// 	}
// 	if x_solved.Data[2].Scalar < 1.9999999999 || x_solved.Data[2].Scalar > 2.0000000001 {
// 		t.Error("Gauss_Jordan_Elimination() failed")
// 	}

// 	/// @notice Testing Gauss_Jordan_Elimination() batched

// 	// Setup the batched version of the above test
// 	A = Zero_Tensor([]int{3, 3, 3}, true)

// 	A.Data[0].Scalar = 2
// 	A.Data[1].Scalar = 4
// 	A.Data[2].Scalar = -2
// 	A.Data[3].Scalar = 4
// 	A.Data[4].Scalar = 9
// 	A.Data[5].Scalar = -3
// 	A.Data[6].Scalar = -2
// 	A.Data[7].Scalar = -3
// 	A.Data[8].Scalar = 7
// 	A.Data[9].Scalar = 2
// 	A.Data[10].Scalar = 4
// 	A.Data[11].Scalar = -2
// 	A.Data[12].Scalar = 4
// 	A.Data[13].Scalar = 9
// 	A.Data[14].Scalar = -3
// 	A.Data[15].Scalar = -2
// 	A.Data[16].Scalar = -3
// 	A.Data[17].Scalar = 7
// 	A.Data[18].Scalar = 2
// 	A.Data[19].Scalar = 4
// 	A.Data[20].Scalar = -2
// 	A.Data[21].Scalar = 4
// 	A.Data[22].Scalar = 9
// 	A.Data[23].Scalar = -3
// 	A.Data[24].Scalar = -2
// 	A.Data[25].Scalar = -3
// 	A.Data[26].Scalar = 7

// 	x = Zero_Tensor([]int{3, 3, 1}, true)

// 	x.Data[0].Scalar = -1
// 	x.Data[1].Scalar = 2
// 	x.Data[2].Scalar = 2
// 	x.Data[3].Scalar = -1
// 	x.Data[4].Scalar = 2
// 	x.Data[5].Scalar = 2
// 	x.Data[6].Scalar = -1
// 	x.Data[7].Scalar = 2
// 	x.Data[8].Scalar = 2

// 	b = MatMul(A, x, true)

// 	// solve for x
// 	x_solved = Gauss_Jordan_Elimination(A, b, true)

// 	// check that the solution is correct (within 1e-10)
// 	if x_solved.Data[0].Scalar+x_solved.Data[3].Scalar+x_solved.Data[6].Scalar < -3.0000000001 || x_solved.Data[0].Scalar+x_solved.Data[3].Scalar+x_solved.Data[6].Scalar > -2.9999999999 {
// 		t.Error("Gauss_Jordan_Elimination() failed")
// 	}
// 	if x_solved.Data[1].Scalar+x_solved.Data[4].Scalar+x_solved.Data[7].Scalar < 5.9999999999 || x_solved.Data[1].Scalar+x_solved.Data[4].Scalar+x_solved.Data[7].Scalar > 6.0000000001 {
// 		t.Error("Gauss_Jordan_Elimination() failed")
// 	}
// 	if x_solved.Data[2].Scalar+x_solved.Data[5].Scalar+x_solved.Data[8].Scalar < 5.9999999999 || x_solved.Data[2].Scalar+x_solved.Data[5].Scalar+x_solved.Data[8].Scalar > 6.0000000001 {
// 		t.Error("Gauss_Jordan_Elimination() failed")
// 	}
// }
