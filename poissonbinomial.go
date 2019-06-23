package statext

import (
	"github.com/argusdusty/gofft"
	"gonum.org/v1/gonum/stat/sampleuv"
	"math"
	"math/rand"
)

// PoissonBinomial represents a random variable whose value is the sum of
// independent Bernoulli trials that are not necessarily identically distributed.
// The value of entries in P must be between 0 and 1.
// More information at https://en.wikipedia.org/wiki/Poisson_binomial_distribution.
type PoissonBinomial struct {
	p   []float64
	dim int
	src rand.Source

	pmf []float64
	cdf []float64
}

// NewPoissonBinomial creates a new Poisson binomial distribution with the given parameters p.
// NewPoissonBinomial will panic if len(p) == 0, or if any p is < 0 or > 1.
func NewPoissonBinomial(p []float64, src rand.Source) PoissonBinomial {
	if len(p) == 0 {
		panic("poisson binomial: zero dimensional input")
	}
	for _, v := range p {
		if v < 0 {
			panic("poisson binomial: prob less than 0")
		} else if v > 1 {
			panic("poisson binomial: prob greater than 1")
		}
	}
	dist := PoissonBinomial{
		p:   p,
		src: src,
	}
	dist.pmf = dist.computePmf()
	dist.cdf = dist.computeCdf()
	return dist
}

// computePmf computes the pmf of the Poisson binomial distribution
// Running time: O(N*log(N)^2)
func (p PoissonBinomial) computePmf() []float64 {
	m := 4 // Starting block size
	N := len(p.p) + 1
	n := gofft.NextPow2(N)          // Number of probability arrays to convolve
	data := make([]complex128, n*m) // Working space
	for i, x := range p.p {
		// Initialize arrays to [1-x, x, 0, 0]
		data[i*m] = complex(1-x, 0)
		data[i*m+1] = complex(x, 0)
	}
	for i := N - 1; i < n; i++ {
		// "zero"-pad out to next power of 2
		// Using arrays of [1, 0, 0, 0]
		data[i*m] = 1
	}
	// Do the FFT convolutions
	err := gofft.FastMultiConvolve(data, m, true)
	if err != nil {
		panic(err)
	}
	pmf := gofft.Complex128ToFloat64Array(data[:N])
	return pmf
}

func (p PoissonBinomial) computeCdf() []float64 {
	cdf := make([]float64, len(p.pmf))
	var t float64
	for i := 0; i < len(p.pmf); i++ {
		t += p.pmf[i]
		cdf[i] = t
	}
	return cdf
}

// CDF computes the value of the cumulative distribution function at x.
func (p PoissonBinomial) CDF(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x <= float64(len(p.p)) {
		return p.cdf[int(x)]
	}
	return 1
}

// ExKurtosis returns the excess kurtosis of the distribution.
func (p PoissonBinomial) ExKurtosis() float64 {
	var exkurtosis float64
	for _, prob := range p.p {
		exkurtosis += (1 - 6*(1-prob)*prob) * (1 - prob) * prob
	}
	exkurtosis /= math.Pow(p.StdDev(), 4)
	return exkurtosis
}

// LogProb computes the natural logarithm of the value of the probability
// density function at x.
func (p PoissonBinomial) LogProb(x float64) float64 {
	if x < 0 || x > float64(len(p.p)) || math.Floor(x) != x {
		return math.Inf(-1)
	}
	return math.Log(p.Prob(x))
}

// Mean returns the mean of the probability distribution.
func (p PoissonBinomial) Mean() float64 {
	var mean float64
	for _, prob := range p.p {
		mean += prob
	}
	return mean
}

// NumParameters returns the number of parameters in the distribution.
func (p PoissonBinomial) NumParameters() int {
	return len(p.p)
}

// Prob computes the value of the probability density function at x.
func (p PoissonBinomial) Prob(x float64) float64 {
	if x < 0 || x > float64(len(p.p)) || math.Floor(x) != x {
		return 0
	}
	return p.pmf[int(x)]
}

// Rand returns a random sample drawn from the distribution.
func (p PoissonBinomial) Rand() float64 {
	idx, _ := sampleuv.NewWeighted(p.pmf, nil).Take()
	return float64(idx)
}

// Skewness returns the skewness of the distribution.
func (p PoissonBinomial) Skewness() float64 {
	var skewness float64
	for _, prob := range p.p {
		skewness += (1 - 2*prob) * (1 - prob) * prob
	}
	skewness /= math.Pow(p.StdDev(), 3)
	return skewness
}

// StdDev returns the standard deviation of the probability distribution.
func (p PoissonBinomial) StdDev() float64 {
	return math.Sqrt(p.Variance())
}

// Survival returns the survival function (complementary CDF) at x.
func (p PoissonBinomial) Survival(x float64) float64 {
	return 1 - p.CDF(x)
}

// Variance returns the variance of the probability distribution.
func (p PoissonBinomial) Variance() float64 {
	var variance float64
	for _, prob := range p.p {
		variance += (1 - prob) * prob
	}
	return variance
}
