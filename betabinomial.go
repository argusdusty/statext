package statext

import (
	"math"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mathext"
	"gonum.org/v1/gonum/stat/combin"
)

// BetaBinomial implements the beta-binomial distribution, a discrete probability distribution
// that expresses the probability of a given number of successful Bernoulli trials
// out of a total of n, each with a random success probability drawn from the Beta distribution
// The binomial distribution has the density function:
//  f(k) = (n choose k) Beta(k+alpha, n-k+beta)/Beta(alpha, beta)
// For more information, see https://en.wikipedia.org/wiki/Beta-binomial_distribution.
type BetaBinomial struct {
	// N is the total number of Bernoulli trials. N must be greater than 0.
	N float64
	// Alpha is the left shape parameter of the beta distribution. Alpha must be greater than 0.
	Alpha float64
	// Beta is the right shape parameter of the beta distribution. Beta must be greater than 0.
	Beta float64

	Src rand.Source
}

// CDF computes the value of the cumulative distribution function at x.
func (b BetaBinomial) CDF(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x >= b.N {
		return 1
	}
	x = math.Floor(x)
	// TODO: faster method?
	var cdf float64
	for i := 0; i <= int(x); i++ {
		cdf += b.Prob(float64(i))
	}
	return cdf
}

// ExKurtosis returns the excess kurtosis of the distribution.
func (b BetaBinomial) ExKurtosis() float64 {
	v := (b.Alpha + b.Beta)
	v2 := b.Alpha * b.Beta
	return v*v*(v+1)/(b.N*v2*(v+2)*(v+3)*(v+b.N))*(v*(v+1+6*b.N)+3*v2*(b.N-2)+6*b.N*b.N-3*v2*b.N*(6-b.N)/v-18*v2*b.N*b.N/(v*v)) - 3
}

// LogProb computes the natural logarithm of the value of the probability
// density function at x.
func (b BetaBinomial) LogProb(x float64) float64 {
	if x < 0 || x > b.N || math.Floor(x) != x {
		return math.Inf(-1)
	}
	lb := combin.LogGeneralizedBinomial(b.N, x)
	return lb + mathext.Lbeta(x+b.Alpha, b.N-x+b.Beta) - mathext.Lbeta(b.Alpha, b.Beta)
}

// Mean returns the mean of the probability distribution.
func (b BetaBinomial) Mean() float64 {
	return b.N * b.Alpha / (b.Alpha + b.Beta)
}

// NumParameters returns the number of parameters in the distribution.
func (BetaBinomial) NumParameters() int {
	return 3
}

// Prob computes the value of the probability density function at x.
func (b BetaBinomial) Prob(x float64) float64 {
	return math.Exp(b.LogProb(x))
}

// Rand returns a random sample drawn from the distribution.
func (b BetaBinomial) Rand() float64 {
	// TODO: faster method?
	r := rand.New(b.Src).Float64()
	for i := 0; i <= int(b.N); i++ {
		r -= b.Prob(float64(i))
		if r < 0 {
			return float64(i)
		}
	}
	return 0.0
}

// Skewness returns the skewness of the distribution.
func (b BetaBinomial) Skewness() float64 {
	v := b.Alpha + b.Beta
	return (v + 2*b.N) * (b.Beta - b.Alpha) / (v + 2) * math.Sqrt((v+1)/(b.N*b.Alpha*b.Beta*(v+b.N)))
}

// StdDev returns the standard deviation of the probability distribution.
func (b BetaBinomial) StdDev() float64 {
	return math.Sqrt(b.Variance())
}

// Survival returns the survival function (complementary CDF) at x.
func (b BetaBinomial) Survival(x float64) float64 {
	return 1 - b.CDF(x)
}

// Variance returns the variance of the probability distribution.
func (b BetaBinomial) Variance() float64 {
	v := b.Alpha + b.Beta
	return b.N * b.Alpha * b.Beta * (v + b.N) / (v * v * (v + 1))
}
