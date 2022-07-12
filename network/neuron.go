package network

import (
	"github.com/deslum/go-neural-network/math"
)

type Neuron struct {
	bias float64
}

func NewNeuron(bias float64) *Neuron {
	return &Neuron{
		bias: bias,
	}
}

func (o *Neuron) FeedForward(input []float64, weights []float64) (feedForward float64, sum float64) {
	var result float64

	for i, weight := range weights {
		result += weight * input[i]
	}

	result += o.bias

	return math.Sigmoid(result), result
}
