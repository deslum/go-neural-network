package math

import (
	"math"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func DerivSigmoid(x float64) float64 {
	fx := Sigmoid(x)
	return fx * (1 - fx)
}

func MSELoss(output, expected []float64) float64 {
	if len(output) != len(expected) {
		return 0
	}

	loss := 0.0
	for i, element := range output {
		loss += math.Pow(element-expected[i], 2)
	}

	loss /= float64(len(expected))

	return loss
}
