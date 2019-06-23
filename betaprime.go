package statext

import (
	"gonum.org/v1/gonum/mathext"
	"math"
	"math/rand"
)

// BetaPrime implements the BetaPrime distribution, a two-parameter
// continuous distribution with support over the positive real numbers.
//
// The beta prime distribution has density function
//  x^(α-1) * (1+x)^(-α-β) * Γ(α+β) / (Γ(α)*Γ(β))
//
// For more information, see https://en.wikipedia.org/wiki/Beta_prime_distribution
type BetaPrime struct {
	// Alpha is the left shape parameter of the distribution. Alpha must be greater
	// than 0.
	Alpha float64
	// Beta is the right shape parameter of the distribution. Beta must be greater
	// than 0.
	Beta float64

	Src rand.Source
}

// CDF computes the value of the cumulative density function at x.
func (b BetaPrime) CDF(x float64) float64 {
	return mathext.RegIncBeta(b.Alpha, b.Beta, x/(1+x))
}

// ExKurtosis returns the excess kurtosis of the distribution.
//
// ExKurtosis returns NaN if the Beta parameter is less or equal to 4.
func (b BetaPrime) ExKurtosis() float64 {
	if b.Beta <= 4 {
		return math.NaN()
	}
	return 6 * (b.Alpha*(b.Alpha+b.Beta-1)*(5*b.Beta-11) + (b.Beta-1)*(b.Beta-1)*(b.Beta-2)) / (b.Alpha * (b.Alpha + b.Beta - 1) * (b.Beta - 3) * (b.Beta - 4))
}

// LogProb computes the natural logarithm of the value of the probability
// density function at x.
func (b BetaPrime) LogProb(x float64) float64 {
	return math.Log(x)*(b.Alpha-1) + math.Log1p(x)*(-b.Alpha-b.Beta) - mathext.Lbeta(b.Alpha, b.Beta)
}

// Mean returns the mean of the probability distribution.
//
// Mean returns NaN if the Beta parameter is less than or equal to 1.
func (b BetaPrime) Mean() float64 {
	if b.Beta <= 1 {
		return math.NaN()
	}
	return b.Alpha / (b.Beta - 1)
}

// Mode returns the mode of the distribution.
//
// Mode returns NaN if the Beta parameter is less than or equal to 1.
func (b BetaPrime) Mode() float64 {
	if b.Beta <= 1 {
		return math.NaN()
	}
	return (b.Alpha - 1) / (b.Beta + 1)
}

// NumParameters returns the number of parameters in the distribution.
func (b BetaPrime) NumParameters() int {
	return 2
}

// Prob computes the value of the probability density function at x.
func (b BetaPrime) Prob(x float64) float64 {
	return math.Exp(b.LogProb(x))
}

// Quantile returns the inverse of the cumulative distribution function.
func (b BetaPrime) Quantile(p float64) float64 {
	if p < 0 || p > 1 {
		panic("beta prime: bad percentile")
	}
	y := mathext.InvRegIncBeta(b.Alpha, b.Beta, p)
	return y / (1 - y)
}

// Rand returns a random sample drawn from the distribution.
func (b BetaPrime) Rand() float64 {
	if b.Src != nil {
		return b.Quantile(rand.New(b.Src).Float64())
	} else {
		return b.Quantile(rand.Float64())
	}
}

// Skewness returns the skewness of the distribution.
//
// Skewness returns NaN if the Beta parameter is less than or equal to 3.
func (b BetaPrime) Skewness() float64 {
	if b.Beta <= 3 {
		return math.NaN()
	}
	return 2 * (2*b.Alpha + b.Beta - 1) / (b.Beta - 3) * math.Sqrt((b.Beta-2)/(b.Alpha*(b.Alpha+b.Beta-1)))
}

// StdDev returns the standard deviation of the probability distribution.
//
// StdDev returns NaN if the Beta parameter is less than or equal to 2.
func (b BetaPrime) StdDev() float64 {
	if b.Beta <= 2 {
		return math.NaN()
	}
	return math.Sqrt(b.Variance())
}

// Survival returns the survival function (complementary CDF) at x.
func (b BetaPrime) Survival(x float64) float64 {
	return 1 - b.CDF(x)
}

// Variance returns the variance of the probability distribution.
//
// Variance returns NaN if the Beta parameter is less than or equal to 2.
func (b BetaPrime) Variance() float64 {
	if b.Beta <= 2 {
		return math.NaN()
	}
	return b.Alpha * (b.Alpha + b.Beta - 1) / ((b.Beta - 2) * (b.Beta - 1) * (b.Beta - 1))
}
