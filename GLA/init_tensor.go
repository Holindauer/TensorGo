package GLA

// This source file contains functions for initializing tensors

// This interface initializes each element of the tensor.Data member
type TensorInitializer interface {
	ValueAt(index int) float64
}

// InitializeData creates and initializes a Tensor's data using a given TensorInitializer
func InitializeData(shape []int, initializer TensorInitializer) *Tensor {
	// create a new Tensor struct, setting its shape
	A := new(Tensor)
	A.Shape = shape
	num_elements := Product(shape) // <--- Product() is a function from utils.go
	A.Data = make([]float64, num_elements)

	for i := range A.Data {
		A.Data[i] = initializer.ValueAt(i) // set each element
	}
	return A
}

//=============================================================================================================Tensors of Constant Values

type ConstInitializer struct{ value float64 }

func (ci *ConstInitializer) ValueAt(index int) float64 { // <-- sets each element
	return ci.value
}

func (ci *ConstInitializer) Execute(shape []int) *Tensor { // <--- Execute() from Batched_Initializer_Operation() in batching.go
	return InitializeData(shape, ci)
}

func Const_Tensor(shape []int, constant float64, batching bool) *Tensor {

	initializer := &ConstInitializer{value: constant} // <-- set cont val in initializer
	if !batching {
		return InitializeData(shape, initializer) // single op
	} else {
		return Batched_Initializer_Operation(initializer, shape) // single op
	}
}

func Zero_Tensor(shape []int, batching bool) *Tensor {
	return Const_Tensor(shape, 0, batching)
}

func Ones_Tensor(shape []int, batching bool) *Tensor {
	return Const_Tensor(shape, 1, batching)
}

//=============================================================================================================Range Tensor

// RangeInitializer sets a value equal to the index for each element
type RangeInitializer struct{}

// This is a method of RangeInitializer. It returns the val at an index of the tensor
func (ri *RangeInitializer) ValueAt(index int) float64 {
	return float64(index)
}

func (ri *RangeInitializer) Execute(shape []int) *Tensor { // <--- Execute() from Batched_Initializer_Operation() in batching.go
	return InitializeData(shape, ri)
}

// populates a tensor's contiguous mem with values from 0 to n-1
func Range_Tensor(shape []int, batching bool) *Tensor {
	if !batching {
		return InitializeData(shape, &RangeInitializer{}) // single op
	} else {
		return Batched_Initializer_Operation(&RangeInitializer{}, shape) // batched op
	}
}

//=============================================================================================================Random Tensors

type RandomInitializer struct {
	min    float64
	max    float64
	random *Random
}

func (ri *RandomInitializer) ValueAt(index int) float64 { // <-- sets each element
	return ri.random.RandInRangeFloat(ri.min, ri.max)
}

func (ri *RandomInitializer) Execute(shape []int) *Tensor { // <--- Execute() from Batched_Initializer_Operation() in batching.go
	return InitializeData(shape, ri)
}

func RandFloat_Tensor(shape []int, lower float64, upper float64, batching bool) *Tensor {

	random := NewRandom() // <--- NewRandom() is a function from utils.go
	ri := &RandomInitializer{min: lower, max: upper, random: random}

	if !batching {
		return InitializeData(shape, ri) // single init
	} else {
		return Batched_Initializer_Operation(ri, shape) // batched init
	}
}

//=============================================================================================================Copy a Tensor

// copy_tensor = tensor.Copy() creates a copy of tensor
func (A *Tensor) Copy() *Tensor {
	B := Zero_Tensor(A.Shape, false)
	copy(B.Data, A.Data) // <--- built in func
	return B
}

//=============================================================================================================Identity Square Matrix

type IdentityInitializer struct {
	size int
}

func (ii *IdentityInitializer) ValueAt(index int) float64 { // <-- sets each element
	// Determine the row and column from the index
	row := index / ii.size
	col := index % ii.size
	if row == col {
		return 1
	}
	return 0
}

func (ii *IdentityInitializer) Execute(shape []int) *Tensor { // <--- Execute() from Batched_Initializer_Operation() in batching.go
	return InitializeData(shape, ii)
}

// Creates Identity Matrix (Only Square Matrices)
func Eye(shape []int, batching bool) *Tensor {

	size := shape[len(shape)-1] // <-- get last elem of shape to avoid getting batch size as mat size
	initializer := &IdentityInitializer{size: size}

	if !batching {
		return InitializeData(shape, initializer) // single tensor init
	} else {
		return Batched_Initializer_Operation(initializer, shape) // batched tensor init
	}
}

//=============================================================================================================Gramien Matrix

type GramienInitializer struct{}

func (gi *GramienInitializer) Execute(A *Tensor) *Tensor { // <--- Execute() from Batch_Tensor_Tensor_Interface() in batching.go
	if len(A.Shape) != 2 {
		panic("Within Gramien_Matrix(): Tensor must be 2D")
	}
	return MatMul(A, A.Transpose([]int{1, 0}), false) // <--- A @ A.T
}

// Gramien_Matrix(A) returns A * A.T for a square matrix.
func Gram(A *Tensor, batching bool) *Tensor {

	GI := &GramienInitializer{}
	if !batching {
		return GI.Execute(A) // single op
	} else {
		return Batch_Tensor_Tensor_Operation(GI, A) // batched op
	}
}
