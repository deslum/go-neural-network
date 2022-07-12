package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/deslum/go-neural-network/math"
	"github.com/deslum/go-neural-network/network"
)

func main() {
	data := [][]float64{
		{54.4, 165.1}, // Alice
		{65.44, 183},  // Bob
		{62.2, 178},   // Charlie
		{49, 152},     // Diana
	}

	minmax := math.NewMinMax()
	data = minmax.Calc(data)

	all_y_trues := []float64{
		1, // Alice
		0, //  Bob
		0, // Charlie
		1, // Diana
	}

	rand.Seed(time.Now().Unix())
	nw := network.NewNetwork()
	nw.Train(data, all_y_trues)

	emily := minmax.Get([]float64{54, 160})
	frank := minmax.Get([]float64{192, 75})

	model, err := nw.SaveModel()
	if err != nil {
		return
	}

	nw = network.NewNetwork()
	err = nw.LoadModel(model)
	if err != nil {
		return
	}

	fmt.Println(nw.FeedForward(emily))
	fmt.Println(nw.FeedForward(frank))
}
