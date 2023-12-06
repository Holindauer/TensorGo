package TG

// // Steps for implementing PCA (matrix operaiton)
// // 1. Normalize the data
// // 2. Compute the covariance matrix
// // 3. Compute the eigenvalues and eigenvectors of the covariance matrix
// // 4. Sort the eigenvalues and their corresponding eigenvectors
// // 5. Select the top k eigenvectors
// // 6. Transform the original data set

// import (
// 	"fmt"
// )

// // here will be define a function for standarization of a dataset. This will be a batched only funciton.
// // standardize_feature = (feature - mean_of_feature) / Standard Deviation of feature

// type Standardize_Operation struct{}

// // By treating standardizardization as a batched operation, we can ensure that the Tensors containing the mean and std line up
// // element by element with the element of the Tensor that is being standardized. This is because the axis op collapesed the 0'th axis
// func (s Standardize_Operation) Execute(A, A_Mean_Axis_0, A_Std_Axis_0 *Tensor) *Tensor {

// 	Standardized_A := Zero_Tensor(A.Shape, false)
// 	indices := make([]int, len(A.Shape)) // <--- to hold a single multi-dimensional indices

// 	// Consider a 3x3x3 tensor. The indices will start at [0, 0, 0], [0, 0, 1], then [0, 0, 2], [0, 1, 0]... etc.
// 	for i := 0; i < len(A.Data); i++ {
// 		// Standarize the current index.
// 		resultIndex := Index(indices, A.Shape) // <--- compute the 1D index of the result tensor

// 		if A_Std_Axis_0.Data[indices[0]] == 0 {
// 			Standardized_A.Data[resultIndex] = 0 // Handle the case where the standard deviation is zero.
// 		} else {
// 			Standardized_A.Data[resultIndex] = (A.Data[resultIndex] - A_Mean_Axis_0.Data[indices[0]]) / A_Std_Axis_0.Data[indices[0]]
// 		}

// 		// Drecrement multi-dimensional indices.
// 		for dim := len(A.Shape) - 1; dim >= 0; dim-- {
// 			indices[dim]++
// 			if indices[dim] < A.Shape[dim] { // break to the next iter if we havemt reached the end of the current dimension
// 				break
// 			}
// 			indices[dim] = 0
// 		}
// 	}

// 	return Standardized_A
// }

// // Standardize() is a batched only method that standardizes each element of a batched tensors using z = (x - mean) / std
// // where x is the element of the batched tensor, and mean and std are the mean and standard deviation of that specific
// // element in the batched tensor.

// // Standardization() works for batched Tensors of arbitrary dimmensionality by first Using Mean_Axis() and Std_Axis() operations
// // to collapse the batch dimmension of the Tensor into their respective statistics. Then, each element is passed to the Execute()
// // Method of the Standardize_Operation struct, which iterates through the mulit-dimensional indices of the Tensor standarizing each
// func (A *Tensor) Standardize() *Tensor {


// 	// In order for this to work we need to stack the batch
// 	A_Mean_Axis_0 := A.Mean_Axis(0, false) // <--- batching set to false because we want to mean each feature
// 	A_Std_Axis_0 := A.Std_Axis(0, false)

// 	fmt.Println("A_Mean_Axis_0.Shape: ", A_Mean_Axis_0.Shape)
// 	Display_Matrix(A_Mean_Axis_0, false)
// 	fmt.Println("A_Std_Axis_0.Shape: ", A_Std_Axis_0.Shape)
// 	Display_Matrix(A_Std_Axis_0, false)

// 	return Batch_ThreeTensor_Tensor_Operation(Standardize_Operation{}, A, A_Mean_Axis_0, A_Std_Axis_0)
// }


// // // This function concatenates all elements of a batched
// // func Stack_Batch(A *Tensor) *Tensor{


// // }