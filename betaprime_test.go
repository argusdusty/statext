package statext

import (
	"math/rand"
	"sort"
	"testing"
)

func TestBetaPrime(t *testing.T) {
	src := rand.New(rand.NewSource(1))
	for i, b := range []BetaPrime{
		{12, 16, src},
		{49, 34, src},
		{51, 67, src},
	} {
		testBetaPrime(t, b, i)
	}
}

func testBetaPrime(t *testing.T, b BetaPrime, i int) {
	const (
		tol  = 1e-2
		n    = 1e6
		bins = 50
	)
	x := make([]float64, n)
	generateSamples(x, b)
	sort.Float64s(x)

	testRandLogProbContinuous(t, i, 0, x, b, tol, bins)
	checkProbContinuous(t, i, x, b, 1e-3)
	checkMean(t, i, x, b, tol)
	checkVarAndStd(t, i, x, b, tol)
	checkExKurtosis(t, i, x, b, 1e-1)
	checkSkewness(t, i, x, b, 5e-2)
	checkQuantileCDFSurvival(t, i, x, b, 5e-3)
}
