package statext

import (
	"math"
	"testing"
)

func TestPoissonBinomialProb(t *testing.T) {
	const tol = 1e-12
	for cas, test := range []struct {
		Dir  *PoissonBinomial
		x    float64
		prob float64
	}{
		{
			NewPoissonBinomial([]float64{1, 1, 1}, nil),
			3,
			1.0,
		},
		{
			NewPoissonBinomial([]float64{0.6, 0.1, 0.8}, nil),
			2,
			0.476,
		},
		{
			NewPoissonBinomial([]float64{0.2, 0.3}, nil),
			0,
			0.56,
		},
		{
			NewPoissonBinomial([]float64{0.2, 0.3}, nil),
			1,
			0.38,
		},
		{
			NewPoissonBinomial([]float64{0.2, 0.3}, nil),
			2,
			0.06,
		},
	} {
		p := test.Dir.Prob(test.x)
		if math.Abs(p-test.prob) > tol {
			t.Errorf("Probablility mismatch. Case %v. Got %v, want %v", cas, p, test.prob)
		}
	}
}

func TestPoissonBinomialCDF(t *testing.T) {
	const tol = 1e-12
	for cas, test := range []struct {
		Dir  *PoissonBinomial
		x    float64
		prob float64
	}{
		{
			NewPoissonBinomial([]float64{1, 1, 1}, nil),
			3,
			1.0,
		},
		{
			NewPoissonBinomial([]float64{0.6, 0.1, 0.8}, nil),
			2,
			0.952,
		},
		{
			NewPoissonBinomial([]float64{0.2, 0.3}, nil),
			0,
			0.56,
		},
		{
			NewPoissonBinomial([]float64{0.2, 0.3}, nil),
			1,
			0.94,
		},
		{
			NewPoissonBinomial([]float64{0.2, 0.3}, nil),
			2,
			1.0,
		},
	} {
		p := test.Dir.CDF(test.x)
		if math.Abs(p-test.prob) > tol {
			t.Errorf("Probablility mismatch. Case %v. Got %v, want %v", cas, p, test.prob)
		}
	}
}
