package main

import (
	"fmt"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

// Example usage and testing of the Value struct and its methods.
func main() {
	// Create Values
	x := NewValue(2, nil, "")
	y := NewValue(3, nil, "")

	// Perform operations
	z := x.Add(y)
	w := z.Mul(y)
	r := w.ReLU()

	r.Backward()

	// Print gradients
	fmt.Println("x.Grad:", x.Grad, "x.Data: ", x.Scalar, "x.Op: ", x.Op)
	fmt.Println("y.Grad:", y.Grad, "y.Data: ", y.Scalar, "y.Op: ", y.Op)
	fmt.Println("z.Grad:", z.Grad, "z.Data: ", z.Scalar, "z.Op: ", z.Op)
	fmt.Println("w.Grad:", w.Grad, "w.Data: ", w.Scalar, "w.Op: ", w.Op)
	fmt.Println("r.Grad:", r.Grad, "r.Data: ", r.Scalar, "r.Op: ", r.Op)

}
