package TG

// import (
// 	. "github.com/Holindauer/Tensor-Go/TensorGo"
// )

/*
* @notice This test checks that the scalar values for MatMul are correctly being processes using the
* Value methods instead of simple float64 multiplication and addition.
 */
// func Test_MatMulGrad_Unbatched(t *testing.T) {

// 	// Create two matmul compatible tensors
// 	noGradA := RangeTensor([]int{5, 6}, false)
// 	noGradB := RangeTensor([]int{6, 5}, false)

// 	// matmul the two tensors
// 	noGradC := MatMul(noGradA, noGradB, false)

// 	// Gradify A and B
// 	GradA := Gradify(noGradA)
// 	GradB := Gradify(noGradB)

// 	// gradient tracked matmul the two tensors
// 	gradC := MatMulGrad(GradA, GradB, false)

// 	// Check that the shapes are correct
// 	if gradC.Shape[0] != noGradC.Shape[0] || gradC.Shape[1] != noGradC.Shape[1] {
// 		t.Errorf("MatMulGrad() failed. Expected Output: %v --- Actual Output: %v", noGradC.Shape, gradC.Shape)
// 	}

// 	for i := 0; i < len(noGradC.Data); i++ {

// 		if gradC.DataReqGrad[i].Scalar != noGradC.Data[i] { // Failing
// 			panic("MatMulGrad() failed. gradC and C are not the same")
// 		} // Check that the values are correct
// 	}
// }
