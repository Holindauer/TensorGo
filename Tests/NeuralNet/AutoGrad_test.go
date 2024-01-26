package TG

import (
	"testing"

	. "github.com/Holindauer/Tensor-Go/TensorGo"
)

/*
* @notice The autodiff implementation is based on andrew karpathy's micrograd, as such the tests
* are also based on his tests.
 */

/*
@notice The following test is adapted from the following test in micrograd:
@dev it should cover all the basic funcionality for an mlp w/ ReLU activation

	def test_sanity_check():

		x = Value(-4.0)
		z = 2 * x + 2 + x
		q = z.relu() + z * x
		h = (z * z).relu()
		y = h + q + q * x
		y.backward()
		xmg, ymg = x, y

		x = torch.Tensor([-4.0]).double()
		x.requires_grad = True
		z = 2 * x + 2 + x
		q = z.relu() + z * x
		h = (z * z).relu()
		y = h + q + q * x
		y.backward()
		xpt, ypt = x, y

		# forward pass went well
		assert ymg.data == ypt.data.item() #should be -20.0
		# backward pass went well
		assert xmg.grad == xpt.grad.item() # should be 46.0
*/
func Test_AutoDiff_SanityCheck(t *testing.T) {

	// Create Values
	x := NewValue(-4.0, nil, "")
	z := x.Mul(NewValue(2.0, nil, "")).Add(NewValue(2.0, nil, "")).Add(x)
	q := z.ReLU().Add(z.Mul(x))
	h := z.Mul(z).ReLU()
	y := h.Add(q).Add(q.Mul(x))

	// Perform backward pass
	y.Backward()

	// Check that the gradients are correct
	if x.Grad != 46.0 {
		t.Errorf("Sanity Check failed. Expected Output: 46.0 --- Actual Output: %v", x.Grad)
	}
}

/*
* @notice This is a test of the Gradify() function. It should convert a Tensor's normal float64 values stored in the Data field
* to Values with the same values stored in the DataReqGrad field.
 */
func Test_Gradify(t *testing.T) {

	A := RangeTensor([]int{2, 5, 5, 2}, true)

	var values []float64
	for i := 0; i < len(A.Data); i++ {
		values = append(values, A.Data[i])
	}

	A_Grad := Gradify(A)

	for i := 0; i < len(A_Grad.DataReqGrad); i++ {

		// grad data should be the same as the original data
		if A_Grad.DataReqGrad[i].Scalar != values[i] {
			t.Errorf("Gradify() failed. Expected Output: %v --- Actual Output: %v", values[i], A_Grad.DataReqGrad[i].Scalar)
		}

		// grad should be zero
		if A_Grad.DataReqGrad[i].Grad != 0.0 {
			t.Errorf("Gradify() failed. Expected Output: 0.0 --- Actual Output: %v", A_Grad.DataReqGrad[i].Grad)
		}

		// Data should be the same as the original data
		if A_Grad.Data[i] != values[i] {
			t.Errorf("Gradify() failed. Expected Output: %v --- Actual Output: %v", values[i], A_Grad.Data[i])
		}
	}

}
