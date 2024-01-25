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
