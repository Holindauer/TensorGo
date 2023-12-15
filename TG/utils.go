package TG

// utils.go contains helper functions for this projects

import (
	"bytes"
	"math/rand"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type Random struct {
	rnd *rand.Rand
}

func NewRandom() *Random {
	source := rand.NewSource(time.Now().UnixNano())
	return &Random{rnd: rand.New(source)}
}

func (r *Random) RandInRangeInt(min, max int) int {
	return min + r.rnd.Intn(max-min)
}

func (r *Random) RandInRangeFloat(min, max float64) float64 {
	return min + r.rnd.Float64()*(max-min)
}

// Helper function for computing the product of elements in a slice
func Product(shape []int) int {
	product := 1
	for _, dim := range shape {
		product *= dim
	}
	return product
}

// This function checks if two tensors are of
// the same shape. It returns a boolean
func Same_Shape(A *Tensor, B *Tensor) bool {

	if len(A.Shape) != len(B.Shape) { // check that they have the same number of dimensions
		return false
	}

	for i := 0; i < len(A.Shape); i++ { // check that each dimension is the same
		if A.Shape[i] != B.Shape[i] {
			return false
		}
	}

	return true
}

// This function is used to create a slice of integer indicies from 0 to n -1 and then have the 0'th and n - 1'th indicies swapped
// This is used to reorder the indicies of a tensor to reorder the contiguous memory of a tensor
func Indicies_First_Last_Swapped(n int) []int {
	indicies := make([]int, n)
	for i := range indicies {
		indicies[i] = i
	}
	indicies[0] = n - 1
	indicies[n-1] = 0
	return indicies
}

// This checks whether the dimmensions of two individual Tensors are compatible for matrix multiplication
func Check_MatMul_Compatibility(A *Tensor, B *Tensor) {

	// check if tensor shapes are compatible for matmul
	if len(A.Shape) != 2 || len(B.Shape) != 2 {
		panic("Within Matmul(): Tensors must both be 2D to compute matmul")
	}

	// check if mxn and nxp
	if A.Shape[1] != B.Shape[0] {
		panic("Within Matmul(): 2D Tensors must be compatible for matmul")
	}
}

// isEqual compares two slices of integers for equality.
func isEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func runPythonScript(scriptName string, args ...string) []float64 {
	// Prepare the command with script name and arguments
	cmdArgs := append([]string{scriptName}, args...)
	cmd := exec.Command("python3", cmdArgs...)

	// Buffers to capture the output
	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	// Run the command and check for errors
	err := cmd.Run()
	if err != nil {
		panic("Within runPythonScript() --- " + err.Error() + " --- Stderr: " + stderr.String())
	}

	// Split the output into separate strings
	output := strings.TrimSpace(out.String())
	strValues := strings.Split(output, " ")

	// Parse each string as a float64
	var result []float64
	for _, str := range strValues {
		value, err := strconv.ParseFloat(str, 64)
		if err != nil {
			panic("Within runPythonScript() --- " + err.Error())
		}
		result = append(result, value)
	}

	return result
}
