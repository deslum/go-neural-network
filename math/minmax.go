package math

type MinMax struct {
	min [2]float64
	max [2]float64
}

func NewMinMax() *MinMax {
	return &MinMax{}
}

func (o *MinMax) Calc(arr [][]float64) [][]float64 {
	for i := 0; i < 2; i++ {
		o.min[i] = arr[0][i]
		o.max[i] = arr[0][i]
	}

	for i := range arr {
		if arr[i][0] <= o.min[0] {
			o.min[0] = arr[i][0]
		}
		if arr[i][0] >= o.max[0] {
			o.max[0] = arr[i][0]
		}
		if arr[i][1] <= o.min[1] {
			o.min[1] = arr[i][1]
		}
		if arr[i][1] >= o.max[1] {
			o.max[1] = arr[i][1]
		}
	}

	for i := range arr {
		arr[i][0] = (arr[i][0] - o.min[0]) / (o.max[0] - o.min[0])
		arr[i][1] = (arr[i][1] - o.min[1]) / (o.max[1] - o.min[1])
	}

	return arr
}

func (o *MinMax) Get(arr []float64) []float64 {
	arr[0] = (arr[0] - o.min[0]) / (o.max[0] - o.min[0])
	arr[1] = (arr[1] - o.min[1]) / (o.max[1] - o.min[1])

	return arr
}
