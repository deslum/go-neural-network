package network

import (
	"encoding/json"
	"fmt"
	"math/rand"

	"github.com/deslum/go-neural-network/math"
)

const (
	learnRate = 0.1
	epochs    = 1000
)

type Network struct {
	neurons []*Neuron
	weights []float64
}

func NewNetwork() *Network {
	var weights = make([]float64, 6)
	for i := 0; i < len(weights); i++ {
		weights[i] = rand.Float64()
	}

	neurons := make([]*Neuron, 3)
	for i := 0; i < 3; i++ {
		neurons[i] = NewNeuron(rand.Float64())
	}

	return &Network{
		neurons: neurons,
		weights: weights,
	}
}

func (o *Network) Train(data [][]float64, allYTries []float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		for iter := range data {
			h1, sumH1 := o.neurons[0].FeedForward(data[iter], []float64{o.weights[0], o.weights[1]})
			h2, sumH2 := o.neurons[1].FeedForward(data[iter], []float64{o.weights[2], o.weights[3]})
			yPred, sumO1 := o.neurons[2].FeedForward([]float64{h1, h2}, []float64{o.weights[4], o.weights[5]})

			// Neuron H1
			o.weights[0] -= learnRate * -2 * (allYTries[iter] - yPred) * o.weights[4] * math.DerivSigmoid(sumO1) * data[iter][0] * math.DerivSigmoid(sumH1)
			o.weights[1] -= learnRate * -2 * (allYTries[iter] - yPred) * o.weights[4] * math.DerivSigmoid(sumO1) * data[iter][1] * math.DerivSigmoid(sumH1)
			o.neurons[0].bias -= learnRate * -2 * (allYTries[iter] - yPred) * o.weights[4] * math.DerivSigmoid(sumO1) * math.DerivSigmoid(sumH1)

			// Neuron H2
			o.weights[2] -= learnRate * -2 * (allYTries[iter] - yPred) * o.weights[5] * math.DerivSigmoid(sumO1) * data[iter][0] * math.DerivSigmoid(sumH2)
			o.weights[3] -= learnRate * -2 * (allYTries[iter] - yPred) * o.weights[5] * math.DerivSigmoid(sumO1) * data[iter][1] * math.DerivSigmoid(sumH2)
			o.neurons[1].bias -= learnRate * -2 * (allYTries[iter] - yPred) * o.weights[5] * math.DerivSigmoid(sumO1) * math.DerivSigmoid(sumH2)

			// Neuron O1
			o.weights[4] -= learnRate * -2 * (allYTries[iter] - yPred) * h1 * math.DerivSigmoid(sumO1)
			o.weights[5] -= learnRate * -2 * (allYTries[iter] - yPred) * h2 * math.DerivSigmoid(sumO1)
			o.neurons[2].bias -= learnRate * -2 * (allYTries[iter] - yPred) * math.DerivSigmoid(sumO1)
		}

		if epoch%1000 == 0 {
			var yPredictions = make([]float64, len(data))
			for i, d := range data {
				yPredictions[i] = o.FeedForward(d)
			}

			loss := math.MSELoss(allYTries, yPredictions)
			fmt.Printf("Epoch %v loss: %v\n", epoch, loss)

		}
	}
}

func (o *Network) FeedForward(data []float64) float64 {
	h1, _ := o.neurons[0].FeedForward(data, []float64{o.weights[0], o.weights[1]})
	h2, _ := o.neurons[1].FeedForward(data, []float64{o.weights[2], o.weights[3]})
	o1, _ := o.neurons[2].FeedForward([]float64{h1, h2}, []float64{o.weights[4], o.weights[5]})
	return o1
}

func (o *Network) SaveModel() ([]byte, error) {
	var biases []float64
	for _, x := range o.neurons {
		biases = append(biases, x.bias)
	}

	return json.Marshal(&Model{
		Biases: biases,
		Weight: o.weights,
	})
}

func (o *Network) LoadModel(b []byte) error {
	var model = new(Model)
	if err := json.Unmarshal(b, &model); err != nil {
		return err
	}

	o.weights = model.Weight
	for i, x := range o.neurons {
		x.bias = model.Biases[i]
	}

	return nil
}
