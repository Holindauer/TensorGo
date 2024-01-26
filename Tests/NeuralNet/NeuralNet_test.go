package TG

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

/*
* @notice This test checks to make sure that all the weight and biases in the MLP
* are initialized correctly to random values between 0 and 1.
 */
func Test_MLP_Init(t *testing.T) {

	//Create a new MLP
	var mlp *Layer = MLP(
		3,                                // input features
		[]int{16, 16, 3},                 // neurons per layer
		[]string{"relu", "relu", "relu"}, // activations
	)

	Summary(mlp)

	numLayers := 0

	for mlp != nil {

		for i := 0; i < len(mlp.Weights.Data); i++ {

			// Check that weights were initialized at all
			if mlp.Weights.DataReqGrad[i].Scalar == 0 {
				t.Errorf("Weights not initialized correctly")
			}

			// Check that weights are between 0 and 1
			if mlp.Weights.DataReqGrad[i].Scalar > 1 || mlp.Weights.DataReqGrad[i].Scalar < 0 {
				t.Errorf("Weights not initialized correctly")
			}
		}

		numLayers++

		mlp = mlp.Next
	}

	if numLayers != 3 {
		t.Errorf("Incorrect number of layers")
	}

}
