package statext

import (
	"gonum.org/v1/gonum/mathext"
	"math"
)

const (
	MAX_DEPTH = 50 // Integrate to at most this depth (should never be reached)
	MIN_DEPTH = 2  // Integrate to at least this depth
)

var cf int
var tcf int

// dirichletWinnerAdaptiveQuadFunc is the function to integrate for the
// dirichlet winner probs.
// Works in-place on result array.
func dirichletWinnerAdaptiveQuadFunc(avgAlpha, y float64, alphas, result, lgammas []float64) {
	if y == 0.0 || y == 1.0 {
		for j := 0; j < len(result); j++ {
			result[j] = 0.0
		}
		return
	}
	x := avgAlpha * y / (1 - y)
	cdfs := 1.0
	var pdf, cdf float64
	for j, alpha := range alphas {
		pdf = math.Exp(math.Log(x)*(alpha-1) - x - lgammas[j])
		//println(alpha, x)
		cdf = mathext.GammaIncReg(alpha, x)
		if cdf == 0.0 {
			result[j] = 0.0
			cdfs = 0.0
			continue
		}
		result[j] = pdf / cdf
		cdfs *= cdf
	}
	for j := 0; j < len(result); j++ {
		result[j] *= avgAlpha * cdfs / ((1 - y) * (1 - y))
	}
}

// dirichletWinnerAdaptiveQuadRecursive implements an adaptive quadrature
// integration over dirichletWinnerAdaptiveQuadFunc.
// Works in-place on result array.
func dirichletWinnerAdaptiveQuadRecursive(tol, avgAlpha, s, e float64, fs, fe, alphas, result, lgammas, ft []float64, depth, min_depth int) {
	n := len(result)
	if depth == MAX_DEPTH {
		for i := 0; i < n; i++ {
			// Average the endpoints and return
			result[i] += (fs[i] + fe[i]) * (e - s) / 2
		}
		//panic("Max depth")
		return
	}
	var Q, Q2 float64
	cf++
	dirichletWinnerAdaptiveQuadFunc(avgAlpha, (s+e)/2, alphas, ft[:n], lgammas)
	for i := 0; i < n; i++ {
		Q = (fs[i] + fe[i]) * (e - s) / 2
		Q2 = (fs[i] + 4*ft[i] + fe[i]) * (e - s) / 6
		if math.Abs(Q-Q2) >= tol || depth < min_depth {
			// Error too large, divide
			dirichletWinnerAdaptiveQuadRecursive(tol, avgAlpha, s, (s+e)/2, fs, ft[:n], alphas, result, lgammas, ft[n:], depth+1, min_depth) // Left-half integration
			dirichletWinnerAdaptiveQuadRecursive(tol, avgAlpha, (s+e)/2, e, ft[:n], fe, alphas, result, lgammas, ft[n:], depth+1, min_depth) // Right-half integration
			return
		}
	}
	// Small enough error, return
	for i := 0; i < n; i++ {
		result[i] += (fs[i] + 4*ft[i] + fe[i]) * (e - s) / 6
	}
}

// DirichletWinner computes the probabilities that each
func DirichletWinner(alphas []float64, tol float64) []float64 {
	cf = 0
	n := len(alphas)
	result := make([]float64, n)
	if n == 1 {
		result[0] = 1.0
		return result
	}
	if n == 2 {
		b := mathext.RegIncBeta(alphas[0], alphas[1], 0.5)
		result[0] = 1.0 - b
		result[1] = b
		return result
	}
	lgammas := make([]float64, n)
	var avgAlpha float64 // Pre-computed average of alpha values, used to speed things up
	for j, alpha := range alphas {
		lgammas[j], _ = math.Lgamma(alpha)
		avgAlpha += alpha
	}
	avgAlpha /= float64(n)
	fs := make([]float64, n)                                            // function result at start point (0.0)
	fe := make([]float64, n)                                            // function result at end point (1.0)
	ft := make([]float64, MAX_DEPTH*n)                                  // Buffer space to use as fs/fe at lower depths
	dirichletWinnerAdaptiveQuadFunc(avgAlpha, 0.0, alphas, fs, lgammas) // Compute start point function result
	dirichletWinnerAdaptiveQuadFunc(avgAlpha, 1.0, alphas, fe, lgammas) // Compute start point function result
	md := MIN_DEPTH
	for {
		dirichletWinnerAdaptiveQuadRecursive(tol, avgAlpha, 0.0, 1.0, fs, fe, alphas, result, lgammas, ft, 0, md)
		var sr float64
		for _, v := range result {
			sr += v
		}
		if math.Abs(sr-1) <= float64(2*n)*tol {
			break
		}
		for i, _ := range result {
			result[i] = 0
		}
		md++
	}
	tcf += cf
	return result
}
