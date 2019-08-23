package statext

import (
	"math"
	"math/rand"
	"testing"
)

func runDirichletWinnerTestCase(a, rt []float64, t *testing.T) {
	for _, tol := range []float64{1e-3, 1e-8, 1e-13} {
		r := DirichletWinner(a, tol)
		for i := 0; i < len(a); i++ {
			if math.Abs(rt[i]-r[i]) > tol {
				t.Errorf("Dirichlet large error: %g target: %g result: %g tol: %g", rt[i]-r[i], rt[i], r[i], tol)
			}
		}
	}
}

// Verifies that the sums of the probabilities is equal to 1.0 (within given tolerance)
func runDirichletWinnerRandTestCase(a []float64, t *testing.T, full bool) {
	var x float64
	tols := []float64{1e-3, 1e-8, 1e-11} // Can't guarantee past 1e-12 or so due to base errors in GammaInc/etc
	if !full {
		tols = []float64{1e-8}
	}
	for _, tol := range tols {
		r := DirichletWinner(a, tol)
		x = 0.0
		for _, v := range r {
			x += v
		}
		if math.Abs(1-x) > float64(2*len(r))*tol {
			t.Fatalf("Dirichlet large sum error: %g target: %g result: %g tol: %g", 1-x, 1., x, tol)
		}
	}
}

func TestDirichletWinner(t *testing.T) {
	testCases := [][2][]float64{
		{{5.5, 10.5, 15.5}, {0.006730827936742794, 0.15691248315301745, 0.83635668891024}},
		{{50.5, 100.5, 150.5}, {1.2913384498578148e-13, 0.0007572193068734463, 0.9992427806931501}},
	}
	for _, testCase := range testCases {
		runDirichletWinnerTestCase(testCase[0], testCase[1], t)
		runDirichletWinnerTestCase(testCase[0], testCase[1], t) // Test to make sure testCase[1] isn't altered
	}
	for i := 0; i < 100; i++ {
		n := rand.Intn(10) + 1
		a := make([]float64, n)
		ta := rand.ExpFloat64() * math.Pow(1.1, float64(i))
		for j := 0; j < n; j++ {
			a[j] = ta + (rand.ExpFloat64() * math.Sqrt(ta))
		}
		runDirichletWinnerRandTestCase(a, t, true)
	}
	for i := 0; i < 100; i++ {
		n := rand.Intn(10) + 1
		a := make([]float64, n)
		for j := 0; j < n; j++ {
			a[j] = rand.ExpFloat64() * math.Pow(1.1, float64(i))
		}
		runDirichletWinnerRandTestCase(a, t, true)
	}
	for i := 0; i < 200; i++ {
		n := rand.Intn(10) + 1
		a := make([]float64, n)
		ta := rand.ExpFloat64() * math.Pow(1.01, float64(i))
		for j := 0; j < n; j++ {
			a[j] = ta + (rand.ExpFloat64() * math.Sqrt(ta))
		}
		runDirichletWinnerRandTestCase(a, t, false)
	}
	for i := 0; i < 200; i++ {
		n := rand.Intn(10) + 1
		a := make([]float64, n)
		for j := 0; j < n; j++ {
			a[j] = rand.ExpFloat64() * math.Pow(1.01, float64(i))
		}
		runDirichletWinnerRandTestCase(a, t, false)
	}
}

func BenchmarkDirichletWinner(b *testing.B) {
	a := []float64{5.5, 10.5, 15.5}
	for i := 0; i < b.N; i++ {
		DirichletWinner(a, 1e-8)
	}
}

func BenchmarkDirichletWinnerParallel(b *testing.B) {
	a := []float64{5.5, 10.5, 15.5}
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			DirichletWinner(a, 1e-8)
		}
	})
}

func BenchmarkDirichletWinnerLowTol(b *testing.B) {
	a := []float64{5.5, 10.5, 15.5}
	for i := 0; i < b.N; i++ {
		DirichletWinner(a, 1e-11)
	}
}

func BenchmarkDirichletWinnerHighTol(b *testing.B) {
	a := []float64{5.5, 10.5, 15.5}
	for i := 0; i < b.N; i++ {
		DirichletWinner(a, 1e-3)
	}
}
