package main

import (
	. "github.com/Holindauer/Tensor-Go/TG"
)

func main() {

	/*
		This example demonstates how to save and load a Tensor to and from a JSON file.
	*/

	// Init a Tensor
	A := RandFloat64_Tensor([]int{2, 3}, 0, 1, false)

	// Save the Tensor to a JSON file
	A.Save_JSON("A.json")

	// Load the Tensor from the JSON file
	B := Load_JSON("A.json")

	// asert that A.Sum_All() == B.Sum_All()
	if A.Sum_All() != B.Sum_All() {
		panic("A != B")
	}

}
