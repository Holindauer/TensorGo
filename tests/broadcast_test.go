package TG

/*
* @notice broadcast_test.go contains tests for functions in broadcast.go
 */
import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TG"
)

func Test_BroadcastAdd(t *testing.T) {

	// Testing Broadcast_Add() by adding a 3x3x3 batched ones tensor to a 3x3 unbached ones tensor
	A := Ones_Tensor([]int{3, 3, 3}, true)
	B := Ones_Tensor([]int{3, 3}, false)

	// Broadcast_Addition pf A onto B
	B_broad_A := B.Broadcast_Add(A)

	if B_broad_A.Sum_All() != 54 {
		t.Errorf("Broadcast_Add() failed. Expected Output: 36 --- Actual Output: %v", B_broad_A.Sum_All())
	}
}

func Test_BroadcastSubtract(t *testing.T) {
	// Testing Broadcast_Subtract() by subtracting a 3x3x3 batched ones tensor to a 3x3 unbached ones tensor
	A := Ones_Tensor([]int{3, 3, 3}, true)
	B := Ones_Tensor([]int{3, 3}, false)

	// Broadcast_Addition pf A onto B
	B_broad_A := B.Broadcast_Subtract(A)

	if B_broad_A.Sum_All() != 0 {
		t.Errorf("Broadcast_Subtract() failed. Expected Output: 0 --- Actual Output: %v", B_broad_A.Sum_All())
	}

}
