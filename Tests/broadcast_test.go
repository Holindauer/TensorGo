package TG

// broadcast_test.go contains tests for functions in broadcast.go

import (
	"fmt"
	"testing"

	. "github.com/Holindauer/Tensor-Go.git/TG"
)

func Test_Broadcasting(t *testing.T) {

	fmt.Print("\n\nNow Testing Functions from broadcast.go\n-------------------------------------")

	// Testing Broadcast_Add()

	fmt.Println("\nTesting Broadcast_Add()...")
	A := Ones_Tensor([]int{3, 3, 3}, true)
	B := Ones_Tensor([]int{3, 3}, false)

	B_broad_A := B.Broadcast_Add(A)

	if B_broad_A.Sum_All() != 54 {
		t.Errorf("Broadcast_Add() failed. Expected Output: 36 --- Actual Output: %v", B_broad_A.Sum_All())
	}

	fmt.Println("Succsess!")

	// Testing Broadcast_Subtract()

	fmt.Print("\nTesting Broadcast_Subtract()...")

	B_broad_A = B.Broadcast_Subtract(A)

	if B_broad_A.Sum_All() != 0 {
		t.Errorf("Broadcast_Subtract() failed. Expected Output: 0 --- Actual Output: %v", B_broad_A.Sum_All())
	}
	fmt.Println("Succsess!")

}
