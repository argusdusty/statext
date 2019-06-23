package statext

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

func bruteForcePoissonBinomialPmf(p []float64) []float64 {
	n := len(p)
	pmf := make([]float64, n+1)
	pmf[0] = 1.0
	tmp := make([]float64, n+1)
	for _, prob := range p {
		copy(tmp, pmf)
		pmf[0] = tmp[0] * (1 - prob)
		for i := 0; i < n; i++ {
			pmf[i+1] = (1-prob)*tmp[i+1] + prob*tmp[i]
		}
	}
	return pmf
}

func TestPoissonBinomialProb(t *testing.T) {
	const tol = 1e-12

	// Check some specific cases
	for cas, test := range []struct {
		Dir  PoissonBinomial
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
		{
			NewPoissonBinomial([]float64{1.0}, nil),
			-1,
			0.0,
		},
		{
			NewPoissonBinomial([]float64{1.0}, nil),
			2,
			0.0,
		},
	} {
		p := test.Dir.Prob(test.x)
		if math.Abs(p-test.prob) > tol {
			t.Errorf("Probablility mismatch. Case %v. Got %v, want %v", cas, p, test.prob)
		}
	}

	// Test random cases
	for i := 0; i < 100; i++ {
		x := make([]float64, 100)
		for j := 0; j < 100; j++ {
			x[j] = rand.Float64()
		}
		p := NewPoissonBinomial(x, nil)
		s := 0.0
		truePmf := bruteForcePoissonBinomialPmf(x)
		for j := 0; j <= 100; j++ {
			prob := p.Prob(float64(j))
			s += prob
			if math.Abs(prob-truePmf[j]) > tol {
				t.Errorf("Probablility mismatch. Case %v. Got %v, want %v", i, prob, truePmf[j])
			}
		}
		if math.Abs(s-1) > tol {
			t.Errorf("Invalid probabilities, does not sum to 1.0. Got %v, diff %v", s, math.Abs(s-1))
		}
	}
}

func TestPoissonBinomialCDF(t *testing.T) {
	const tol = 1e-12
	for cas, test := range []struct {
		Dir  PoissonBinomial
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
		{
			NewPoissonBinomial([]float64{1.0}, nil),
			-1,
			0.0,
		},
		{
			NewPoissonBinomial([]float64{1.0}, nil),
			2,
			1.0,
		},
	} {
		p := test.Dir.CDF(test.x)
		if math.Abs(p-test.prob) > tol {
			t.Errorf("Probablility mismatch. Case %v. Got %v, want %v", cas, p, test.prob)
		}
	}

	// Test random cases
	for i := 0; i < 100; i++ {
		x := make([]float64, 100)
		for j := 0; j < 100; j++ {
			x[j] = rand.Float64()
		}
		p := NewPoissonBinomial(x, nil)
		truePmf := bruteForcePoissonBinomialPmf(x)
		trueCdf := 0.0
		for j := 0; j <= 100; j++ {
			prob := p.CDF(float64(j))
			trueCdf += truePmf[j]
			if math.Abs(prob-trueCdf) > tol {
				t.Errorf("Probablility mismatch. Case %v. Got %v, want %v", i, prob, trueCdf)
			}
		}
	}
}

func TestPoissonBinomial(t *testing.T) {
	// Check some specific cases
	for i, p := range []PoissonBinomial{
		NewPoissonBinomial([]float64{0.6, 0.1, 0.8}, nil),
		NewPoissonBinomial([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, nil),
	} {
		if len(p.p) != p.NumParameters() {
			t.Errorf("NumParameters mismatch. Case %v. Got %v, want %v", i, p.NumParameters(), len(p.p))
		}
		if !(math.IsInf(p.LogProb(-1), -1) && math.IsInf(p.LogProb(0.5), -1) && math.IsInf(p.LogProb(1000000), -1)) {
			t.Errorf("LogProb out-of-bounds mismatch. Case %v. Got %v, want %v", i, p.LogProb(-1), math.Inf(-1))
		}
		testPoissonBinomial(t, p, i)
	}
}

func testPoissonBinomial(t *testing.T, p PoissonBinomial, i int) {
	const (
		tol = 1e-2
		n   = 2e6
	)
	x := make([]float64, n)
	generateSamples(x, p)
	sort.Float64s(x)

	checkMean(t, i, x, p, tol)
	checkSkewness(t, i, x, p, tol)
	checkVarAndStd(t, i, x, p, tol)
	checkExKurtosis(t, i, x, p, tol)
	checkProbDiscrete(t, i, x, p, tol)
	checkCDFSurvival(t, i, x, p, tol)
}

func randProbs(n int) []float64 {
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = rand.Float64()
	}
	return x
}

func BenchmarkPoissonBinomial(b *testing.B) {
	for _, bm := range []struct {
		size int
		name string
	}{
		{4, "Tiny (4)"},
		{64, "Small (64)"},
		{1024, "Medium (1024)"},
		{16384, "Large (16384)"},
		{262144, "Huge (262144)"},
	} {
		x := randProbs(bm.size)
		b.Run(bm.name, func(b *testing.B) {
			b.SetBytes(int64(bm.size * 8))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				NewPoissonBinomial(x, nil)
			}
		})
	}
}
