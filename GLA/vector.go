package GLA

// vector.go contains functions related to vector/1D Tensor operations

import (
	//"fmt"
	"math"
)

// this helper function checks if two Tensors are vectors of the same dimension
func Same_Dimmension_Vectors(t1 *Tensor, t2 *Tensor) bool {

	if len(t1.Shape) != 1 || len(t2.Shape) != 1 { // check if tensors are vectors (1D)
		return false
	}
	if len(t1.Data) != len(t2.Data) { // check if vectors are of same length
		return false
	}
	return true
}

//---------------------------------------------------------------------------------------------------------------------------- Dot()

type Batched_Dot struct{}

func (op Batched_Dot) Execute(A *Tensor, B *Tensor) *Tensor {

	if Same_Dimmension_Vectors(A, B) == false {
		panic("Within dot(): Tensors must both be vectors to compute dot product")
	}

	var dot float64
	for i := 0; i < len(A.Data); i++ {
		dot += A.Data[i] * B.Data[i]
	}

	// create a tensor with one element to store the dot product
	dot_tensor := Zero_Tensor([]int{1}, false)
	dot_tensor.Data[0] = dot
	return dot_tensor
}

// Dot() computes the dot product of two tensors, returning
// a single element tensor. There is optional batching.
func Dot(A *Tensor, B *Tensor, batching bool) *Tensor {
	if batching {
		return Batch_TwoTensor_Tensor_Operation(Batched_Dot{}, A, B) // batched op
	}
	return Batched_Dot{}.Execute(A, B) // single op
}

//---------------------------------------------------------------------------------------------------------------------------- Norm()

type Batched_Norm struct{}

func (op Batched_Norm) Execute(A *Tensor) *Tensor {
	if len(A.Shape) != 1 {
		panic("Within Norm(): Tensor must be a vector to compute norm")
	}

	normTensor := Zero_Tensor([]int{1}, false)
	normTensor.Data[0] = math.Sqrt(Dot(A, A, false).Data[0])

	return normTensor
}

// Norm() comptues the norm of a vector Tensor into a single element Tensor. There is optional batching.
func Norm(A *Tensor, batching bool) *Tensor {

	if batching {
		return Batch_Tensor_Tensor_Operation(Batched_Norm{}, A) // batched op
	}
	return Batched_Norm{}.Execute(A) // single op
}

//---------------------------------------------------------------------------------------------------------------------------- Unit()

type Batched_Unit struct{}

func (op Batched_Unit) Execute(A *Tensor) *Tensor {
	if len(A.Shape) != 1 {
		panic("Within Unit(): Tensor must be a vector to compute unit vector")
	}

	norm := Norm(A, false).Data[0] // <-- single element Tensor

	if norm == 0 { // handle the case where normalization involves division by zero
		return Zero_Tensor(A.Shape, false)
	}

	// compute the unit vector of A
	Unit_A := Zero_Tensor(A.Shape, false)
	for i := 0; i < len(A.Data); i++ {
		Unit_A.Data[i] = A.Data[i] / norm
	}

	return Unit_A
}

// This function computes the unit vector of a vector Tensor. There is optional batching.
func Unit(A *Tensor, batching bool) *Tensor {
	if batching {
		return Batch_Tensor_Tensor_Operation(Batched_Unit{}, A) // batched op
	}
	return Batched_Unit{}.Execute(A) // single op
}

//---------------------------------------------------------------------------------------------------------------------------- Cosine_Similarity()

type Batched_Cosine_Similarity struct{}

func (op Batched_Cosine_Similarity) Execute(A *Tensor, B *Tensor) *Tensor {
	if Same_Dimmension_Vectors(A, B) == false {
		panic("Within Cosine_Similarity(): Tensors must both be vectors to compute cosine similarity")
	}

	similarityTensor := Zero_Tensor([]int{1}, false)

	normA := Norm(A, false).Data[0]
	normB := Norm(B, false).Data[0]

	if normA == 0 || normB == 0 { // Handle the zero norm case
		similarityTensor.Data[0] = 0 // Or NaN, or any other value you deem appropriate
	} else {
		similarityTensor.Data[0] = Dot(A, B, false).Data[0] / (normA * normB)
	}

	return similarityTensor
}

// This function computes the cosine similarity of two vectors, returning
// a single element Tensor with the answer. There is optional batching.
func Cosine_Similarity(A *Tensor, B *Tensor, batching bool) *Tensor {

	if batching {
		return Batch_TwoTensor_Tensor_Operation(Batched_Cosine_Similarity{}, A, B) // batched op
	}
	return Batched_Cosine_Similarity{}.Execute(A, B) // single op
}

//---------------------------------------------------------------------------------------------------------------------------- Angle()

func Angle_Vector(A *Tensor, B *Tensor, batching bool) *Tensor {

	// anonymous function to apply arccos to a tensor
	applyArcCos := func(A *Tensor) *Tensor {
		for i := 0; i < len(A.Data); i++ {
			A.Data[i] = math.Acos(A.Data[i])
		}
		return A
	}

	if batching {
		return applyArcCos(Cosine_Similarity(A, B, true)) // batched op
	}
	return applyArcCos(Cosine_Similarity(A, B, false)) // single op
}

//---------------------------------------------------------------------------------------------------------------------------- Funcs to check vector features --- which return bools

type Batched_Check_Vectors struct {
	checker func() bool // function to check something about angle between vbectors
}

// This method of the Batched_Check_Vectors struct is callable alone or by using a batching.go function. It operates on single Tensors
// at a time and serves as a way to generalize functions for checking yes/no features about vectors. The checker function used within is passed
// in as a parameter to the Batched_Check_Orthoganal struct before use.
func (op Batched_Check_Vectors) Execute(A *Tensor, B *Tensor) *Tensor {

	if len(A.Shape) != 1 || len(B.Shape) != 1 {
		panic("Within Batched_Check_Vectors(): Tensors must both be vectors to check if perpendicular")
	}

	// Initialize boolTensor with a shape of [1] and boolData slice of length 1
	boolTensor := Zero_Tensor([]int{1}, false)
	boolTensor.BoolData = make([]bool, 1) // Initializing the boolData slice

	// check if the dot product is zero
	if op.checker() {
		boolTensor.BoolData[0] = true
	} else {
		boolTensor.BoolData[0] = false
	}

	return boolTensor
}

// This function is called within the below functions that start with "Check_". It is a generalization of the
// process of instantiating the logic for batched ops and executing them for vector operations that return a bool.
func Check_Vector_Feature(A *Tensor, B *Tensor, checker func() bool, batching bool) *Tensor {

	op := Batched_Check_Vectors{checker: checker} // create op

	if batching {
		return Batch_TwoTensor_Tensor_Operation(op, A, B) // batched op
	}
	return Batched_Check_Vectors{}.Execute(A, B) // single op
}

// Check_Orthogonal() checks if two vectors are orthogonal (w/ optional batching). It passes a checker
// function into the Check_Vector_Feature() generalization, which returns a Tensor of booleans.
func Check_Orthogonal(A *Tensor, B *Tensor, batching bool) *Tensor {

	checker_func := func() bool {
		if Dot(A, B, false).Data[0] == 0 { // dot == 0 indicates orthogonality
			return true
		}
		return false
	}
	// pass in the checker function to Check_Angle_Feature()
	return Check_Vector_Feature(A, B, checker_func, batching)
}

// Check_Acute() checks if two vectors are orthogonal (w/ optional batching). It passes a checker
// function into the Check_Vector_Feature() generalization, which returns a Tensor of booleans.
func Check_Acute(A *Tensor, B *Tensor, batching bool) *Tensor {

	// anon func for within Batched_Check_Orthoganal.Execute()
	checker_func := func() bool {
		if Dot(A, B, false).Data[0] > 0 { // dot > 0 indicates acute angle
			return true
		}
		return false
	}

	return Check_Vector_Feature(A, B, checker_func, batching)
}

// Check_Obtuse() checks if two vectors are orthogonal (w/ optional batching). It passes a checker
// function into the Check_Vector_Feature() generalization, which returns a Tensor of booleans.
func Check_Obtuse(A *Tensor, B *Tensor, batching bool) *Tensor {

	// anon func for within Batched_Check_Orthoganal.Execute()
	checker_func := func() bool {
		if Dot(A, B, false).Data[0] > 0 { // dot < 0 indicates Obtuse angle
			return true
		}
		return false
	}

	return Check_Vector_Feature(A, B, checker_func, batching)
}

//---------------------------------------------------------------------------------------------------------------------------- Outer()

type Batched_Outer struct{}

func (op Batched_Outer) Execute(A *Tensor, B *Tensor) *Tensor {
	// check if tensors are vectors
	if !(len(A.Shape) == 1 && len(B.Shape) == 1) {
		panic("Within Outer_Product(): Tensors must both be vectors to compute outer product")
	}

	// add singletons to the end A's Shape and the beggining of B's Shape
	A = A.Add_Singleton()
	B.Shape = append([]int{1}, B.Shape...)
	Outer := MatMul(A, B, false)

	return Outer
}

// This function computes the outer product of two vectors
// it returns a pointer to a new 2D tensor
func Outer(A *Tensor, B *Tensor, batching bool) *Tensor {

	if batching {
		return Batch_TwoTensor_Tensor_Operation(Batched_Outer{}, A, B) // batched op
	}
	return Batched_Outer{}.Execute(A, B) // single op
}


