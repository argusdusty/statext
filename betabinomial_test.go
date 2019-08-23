package statext

import (
	"golang.org/x/exp/rand"
	"sort"
	"testing"
)

func TestBetaBinomial(t *testing.T) {
	src := rand.New(rand.NewSource(1))
	for i, b := range []BetaBinomial{
		{12, 16, 20, src},
		{49, 34, 34, src},
		{51, 67, 49, src},
	} {
		testBetaBinomial(t, b, i)
	}
}

func testBetaBinomial(t *testing.T, b BetaBinomial, i int) {
	const (
		tol  = 1e-2
		n    = 1e6
		bins = 50
	)
	x := make([]float64, n)
	generateSamples(x, b)
	sort.Float64s(x)

	checkMean(t, i, x, b, tol)
	checkSkewness(t, i, x, b, tol)
	checkVarAndStd(t, i, x, b, tol)
	checkExKurtosis(t, i, x, b, tol)
	checkProbDiscrete(t, i, x, b, tol)
	checkCDFSurvival(t, i, x, b, tol)
}
