package math

import "testing"

func TestDerivSigmoid(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		want float64
	}{
		{
			name: "Case 1: Negative value",
			x:    -1,
			want: 0.19661193324148185,
		},
		{
			name: "Case 2: Positive value",
			x:    1,
			want: 0.19661193324148185,
		},
		{
			name: "Case 3: Zero",
			x:    0,
			want: 0.25,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DerivSigmoid(tt.x); got != tt.want {
				t.Errorf("DerivSigmoid() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMSELoss(t *testing.T) {
	tests := []struct {
		name     string
		output   []float64
		expected []float64
		want     float64
	}{
		{
			name:     "Case1: Integer values",
			output:   []float64{34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24},
			expected: []float64{37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23},
			want:     5.916666666666667,
		},
		{
			name:     "Case2: Float values",
			output:   []float64{0.34, 0.37, 0.44, 0.47, 0.48, 0.48, 0.46, 0.43, 0.32, 0.27, 0.26, 0.24},
			expected: []float64{0.37, 0.40, 0.46, 0.44, 0.46, 0.50, 0.45, 0.44, 0.34, 0.30, 0.22, 0.23},
			want:     0.0005916666666666664,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MSELoss(tt.output, tt.expected); got != tt.want {
				t.Errorf("MSELoss() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		want float64
	}{
		{
			name: "Case1: Positive value",
			x:    0.5,
			want: 0.6224593312018545646389,
		},
		{
			name: "Case2: Negative value",
			x:    -0.5,
			want: 0.3775406687981454,
		},
		{
			name: "Case3: Zero value",
			x:    0,
			want: 0.5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Sigmoid(tt.x); got != tt.want {
				t.Errorf("Sigmoid() = %v, want %v", got, tt.want)
			}
		})
	}
}
