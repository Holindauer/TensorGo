package TG

/*
* @notice init_tensor_test.go contains tests for functions in init_tensor.go
 */
import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

func Test_Ones_Init(t *testing.T) {

	/// @notice Testing Ones_Tensor() Unbatched
	if OnesTensor([]int{2, 3, 4}, false).Sum_All() != 24 {
		t.Errorf("Ones_Tensor() failed. Expected Output: 24 --- Actual Output: %v", OnesTensor([]int{2, 3, 4}, false).Sum_All())
	}

	// @notice Testing Ones_Tensor() Batched
	if OnesTensor([]int{2, 3, 4}, true).Sum_All() != 24 {
		t.Errorf("Ones_Tensor() failed. Expected Output: 24 --- Actual Output: %v", OnesTensor([]int{2, 3, 4}, true).Sum_All())
	}
}

func Test_Zeros_Init(t *testing.T) {

	/// @notice Testing Zero_Tensor() Unbatched

	if ZeroTensor([]int{2, 3, 4}, false).Sum_All() != 0 {
		t.Errorf("Zero_Tensor() failed. Expected Output: 0 --- Actual Output: %v", ZeroTensor([]int{2, 3, 4}, false).Sum_All())
	}

	// @notice Testing Zero_Tensor() Batched
	if ZeroTensor([]int{2, 3, 4}, true).Sum_All() != 0 {
		t.Errorf("Zero_Tensor() failed. Expected Output: 0 --- Actual Output: %v", ZeroTensor([]int{2, 3, 4}, true).Sum_All())
	}
}

func Test_Const_Init(t *testing.T) {

	/// @notice Testing Const_Tensor() Unbatched
	if ConstTensor([]int{2, 3, 4}, 5, false).Sum_All() != 120 {
		t.Errorf("Const_Tensor() failed. Expected Output: 120 --- Actual Output: %v", ConstTensor([]int{2, 3, 4}, 5, false).Sum_All())
	}

	// @notice Testing Const_Tensor() Batched
	if ConstTensor([]int{2, 3, 4}, 5, true).Sum_All() != 120 {
		t.Errorf("Const_Tensor() failed. Expected Output: 120 --- Actual Output: %v", ConstTensor([]int{2, 3, 4}, 5, true).Sum_All())
	}
}

func Test_Range_Init(t *testing.T) {

	/// @notice Test Range_Tensor() Unbatched
	if RangeTensor([]int{3, 3, 3}, false).Sum_All() != 351 {
		t.Errorf("Range_Tensor() failed. Expected Output: 351 --- Actual Output: %v", RangeTensor([]int{3, 3, 3}, false).Sum_All())
	}

	/// @notice Test Range_Tensor() Batched
	if RangeTensor([]int{3, 3, 3}, true).Sum_All() != 108 {
		t.Errorf("Range_Tensor() failed. Expected Output: 108 --- Actual Output: %v", RangeTensor([]int{3, 3, 3}, true).Sum_All())
	}
}

func Test_RandFloat_Init(t *testing.T) {

	// TODO: better test coverage for RandFloat_Tensor()

	// // Test RandFloat_Tensor() <--- Think of a better way to test this Use Standard Deviation?
	// fmt.Print("Testing RandFloat64_Tensor() Unbatched...")
	// rand_float_sum := RandFloat64_Tensor([]int{3, 3, 3}, 0, 1, false).Sum_All()
	// if rand_float_sum > 17 || rand_float_sum < 10 {
	// 	t.Errorf("RandFloat_Tensor() failed. Expected Output Sum Range: 10-17 --- Actual Output Sum: %v", rand_float_sum)
	// }
	// fmt.Println("Succsess!")

	// fmt.Print("Testing RandFloat64_Tensor() Batched...")
	// rand_float_sum = RandFloat64_Tensor([]int{3, 3, 3}, 0, 1, true).Sum_All()
	// if rand_float_sum < 8 || rand_float_sum < 12 {
	// 	t.Errorf("RandFloat_Tensor() failed. Expected Output Sum Range: 8-12 --- Actual Output Sum: %v", rand_float_sum)
	// }
	// fmt.Println("Succsess!")
}

func Test_Copy(t *testing.T) {

	// Test Copy()
	A := RangeTensor([]int{10, 12, 14}, false)
	B := A.Copy()
	if A.Sum_All() != B.Sum_All() {
		t.Errorf("Copy() failed. Expected Output: %v --- Actual Output: %v", A.Sum_All(), B.Sum_All())
	}
}

func Test_Eye(t *testing.T) {
	/// @notice Test Eye() Unbatched
	A := Eye([]int{3, 3}, false)
	if A.Get([]int{0, 0}) != 1 || A.Get([]int{1, 1}) != 1 || A.Get([]int{2, 2}) != 1 {
		t.Errorf("Eye() failed. Expected Output: 1 --- Actual Output: %v", A.Get([]int{0, 0}))
	}

	/// @notice Test Eye() Batched
	A = Eye([]int{3, 3, 3}, true)
	A_Extracted := A.GetBatchElement(1)
	if A_Extracted.Get([]int{0, 0}) != 1 || A_Extracted.Get([]int{1, 1}) != 1 || A_Extracted.Get([]int{2, 2}) != 1 {
		t.Errorf("Eye() failed. Expected Output: 1 --- Actual Output: %v", A.Get([]int{0, 0}))
	}
}
