# statext [![GoDoc][godoc-badge]][godoc] [![Build Status][travis-ci-badge]][travis-ci] [![Report Card][report-card-badge]][report-card]
Extra probability distributions and statistics utilities for Go

Package statext provides additional probability distributions in gonum format, and statistics utilities.

Currently, this library has:
- PoissonBinomial: The Poisson binomial distribution, implemented based on gonum's Dirichlet and Binomial distributions. See [Wikipedia](https://en.wikipedia.org/wiki/Poisson_binomial_distribution) for more info. Uses a custom hierarchical FFT algorithm to efficiently compute the probabilities in O(n\*ln(n)<sup>2</sup>) time, from [gofft](https://github.com/argusdusty/gofft).
- DirichletWinner: A function to compute the probabilities that each output will be the largest when randomly sampling a Dirichlet distribution. Uses a custom adaptive quadrature integration method to efficiently compute the probabilities within a specified tolerance.
- BetaPrime: The Beta prime distribution. See [Wikipedia](https://en.wikipedia.org/wiki/Beta_prime_distribution) for more info.
- BetaBinomial: The Beta-binomial distribution. See [Wikipedia](https://en.wikipedia.org/wiki/Beta-binomial_distribution) for more info.

More is planned.

## License
Original code is licensed under the MIT License found in the LICENSE file. Portions of the code are subject to the additional licenses found in THIRD_PARTY_LICENSES. All third party code is licensed either under a BSD or MIT license.

[travis-ci-badge]:   https://api.travis-ci.org/argusdusty/statext.svg?branch=master
[travis-ci]:         https://api.travis-ci.org/argusdusty/statext
[godoc-badge]:       https://godoc.org/github.com/argusdusty/statext?status.svg
[godoc]:             https://godoc.org/github.com/argusdusty/statext
[report-card-badge]: https://goreportcard.com/badge/github.com/argusdusty/statext
[report-card]:       https://goreportcard.com/report/github.com/argusdusty/statext