package TG

// init_tensor_test.go contains tests for functions in init_tensor.go

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func Test_Tensor_Init(t *testing.T) {

	fmt.Println("\n\nNow Testing Functions from tensor_init.go\n-------------------------------------")

	// Test Zero_Tensor(), Ones_Tensor(), and Const_Tensor()
	fmt.Print("Testing Zero_Tensor() Unbatched...")
	if Zero_Tensor([]int{2, 3, 4}, false).Sum_All() != 0 {
		t.Errorf("Zero_Tensor() failed. Expected Output: 0 --- Actual Output: %v", Zero_Tensor([]int{2, 3, 4}, false).Sum_All())
	}
	fmt.Println("Succsess!")

	fmt.Print("Testing Zero_Tensor() Batched...")
	if Zero_Tensor([]int{2, 3, 4}, true).Sum_All() != 0 {
		t.Errorf("Zero_Tensor() failed. Expected Output: 0 --- Actual Output: %v", Zero_Tensor([]int{2, 3, 4}, true).Sum_All())
	}
	fmt.Println("Succsess!")

	fmt.Print("Testing Ones_Tensor() Unbatched...")
	if Ones_Tensor([]int{2, 3, 4}, false).Sum_All() != 24 {
		t.Errorf("Ones_Tensor() failed. Expected Output: 24 --- Actual Output: %v", Ones_Tensor([]int{2, 3, 4}, false).Sum_All())
	}
	fmt.Println("Succsess!")

	fmt.Print("Testing Ones_Tensor() Batched...")
	if Ones_Tensor([]int{2, 3, 4}, true).Sum_All() != 24 {
		t.Errorf("Ones_Tensor() failed. Expected Output: 24 --- Actual Output: %v", Ones_Tensor([]int{2, 3, 4}, true).Sum_All())
	}
	fmt.Println("Succsess!")

	fmt.Print("Testing Const_Tensor() Unbatched...")
	if Const_Tensor([]int{2, 3, 4}, 5, false).Sum_All() != 120 {
		t.Errorf("Const_Tensor() failed. Expected Output: 120 --- Actual Output: %v", Const_Tensor([]int{2, 3, 4}, 5, false).Sum_All())
	}
	fmt.Println("Succsess!")

	fmt.Print("Testing Const_Tensor() Batched...")
	if Const_Tensor([]int{2, 3, 4}, 5, true).Sum_All() != 120 {
		t.Errorf("Const_Tensor() failed. Expected Output: 120 --- Actual Output: %v", Const_Tensor([]int{2, 3, 4}, 5, true).Sum_All())
	}
	fmt.Println("Succsess!")

	// Test Range_Tensor()
	fmt.Print("Testing Range_Tensor()...")
	if Range_Tensor([]int{3, 3, 3}, false).Sum_All() != 351 {
		t.Errorf("Range_Tensor() failed. Expected Output: 351 --- Actual Output: %v", Range_Tensor([]int{3, 3, 3}, false).Sum_All())
	}
	fmt.Println("Succsess!")

	fmt.Print("Testing Range_Tensor() Batched...")
	if Range_Tensor([]int{3, 3, 3}, true).Sum_All() != 108 {
		t.Errorf("Range_Tensor() failed. Expected Output: 108 --- Actual Output: %v", Range_Tensor([]int{3, 3, 3}, true).Sum_All())
	}
	fmt.Println("Succsess!")

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

	// Test Copy()
	fmt.Print("Testing Copy()...")
	A := Range_Tensor([]int{10, 12, 14}, false)
	B := A.Copy()
	if A.Sum_All() != B.Sum_All() {
		t.Errorf("Copy() failed. Expected Output: %v --- Actual Output: %v", A.Sum_All(), B.Sum_All())
	}
	fmt.Println("Succsess!")

	// Test Eye()

	// Test Eye() Unbatched
	fmt.Print("Testing Eye() Unbatched...")
	A = Eye([]int{3, 3}, false)
	if A.Get([]int{0, 0}) != 1 || A.Get([]int{1, 1}) != 1 || A.Get([]int{2, 2}) != 1 {
		t.Errorf("Eye() failed. Expected Output: 1 --- Actual Output: %v", A.Get([]int{0, 0}))
	}
	fmt.Println("Succsess!")

	// Test Eye() Batched
	fmt.Print("Testing Eye() Batched...")
	A = Eye([]int{3, 3, 3}, true)
	A_Extracted := A.GetBatchElement(1)
	if A_Extracted.Get([]int{0, 0}) != 1 || A_Extracted.Get([]int{1, 1}) != 1 || A_Extracted.Get([]int{2, 2}) != 1 {
		t.Errorf("Eye() failed. Expected Output: 1 --- Actual Output: %v", A.Get([]int{0, 0}))
	}
	fmt.Println("Succsess!")

	// Test Gram()

	// Test Gram() Unbatched
	fmt.Print("Testing Gram() Unbatched...")
	A = Range_Tensor([]int{3, 3}, false)
	B = A.Gram(false)

	if B.Get([]int{0, 0}) != 5 || B.Get([]int{1, 1}) != 50 || B.Get([]int{2, 2}) != 149 {
		t.Errorf("Gram() failed. Expected Output: 5, 14, 23 --- Actual Output: %v, %v, %v", B.Get([]int{0, 0}), B.Get([]int{1, 1}), B.Get([]int{2, 2}))
	}

	fmt.Println("Succsess!")

	// Test Gram() Batched
	fmt.Print("Testing Gram() Batched...")
	A = Range_Tensor([]int{3, 3, 3}, true)
	B = A.Gram(true)
	B_Extracted := B.GetBatchElement(1)

	if B_Extracted.Get([]int{0, 0}) != 5 || B_Extracted.Get([]int{1, 1}) != 50 || B_Extracted.Get([]int{2, 2}) != 149 {
		t.Errorf("Gram() failed. Expected Output: 5, 14, 23 --- Actual Output: %v, %v, %v", B.Get([]int{0, 0}), B.Get([]int{1, 1}), B.Get([]int{2, 2}))
	}

	fmt.Println("Succsess!")

}
