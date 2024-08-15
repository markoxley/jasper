// errorfunctions.go - Error functions used in the neural network.
//
// # Copyright 2024 Mark Oxley
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package jasper

import "math"

// ErrorFunction represents the type of error function used in the neural network.
type ErrorFunction int

const (
	// MeanSquaredError represents the mean squared error function.
	MeanSquaredError ErrorFunction = iota
	// MeanAbsoluteError represents the mean absolute error function.
	MeanAbsoluteError
	// BinaryCrossEntropy represents the binary cross entropy function.
	BinaryCrossEntropy
	// CategoricalCrossEntropy represents the categorical cross entropy function.
	CategoricalCrossEntropy
)

// errorSolver represents the interface for error calculation functions.
type errorSolver interface {
	// e calculates the error between the predicted values and the target values.
	//
	// vs: the predicted values.
	// tgts: the target values.
	// Returns the calculated error.
	e(vs, tgts []float64) float64
}

// getErrorFunction returns the error function corresponding to the given name.
//
// name: the name of the error function.
// Returns the error function corresponding to the given name.
func getErrorFunction(name ErrorFunction) errorSolver {
	switch name {
	case MeanSquaredError:
		return emse{}
	case MeanAbsoluteError:
		return emae{}
	case BinaryCrossEntropy:
		return ebce{}
	case CategoricalCrossEntropy:
		return ecce{}
	}
	return nil
}

// emse represents the mean squared error function.
type emse struct{}

// e calculates the mean squared error between the predicted values and the target values.
//
// vs: the predicted values.
// tgts: the target values.
// Returns the calculated mean squared error.
func (emse) e(vs, tgts []float64) float64 {
	var sum float64 // Initialize the sum to 0

	for i, v := range vs {
		sum += math.Pow(v-tgts[i], 2)
	} // Return the square root of the sum
	return sum / float64(len(vs))
}

// emae represents the mean absolute error function.
type emae struct{}

// e calculates the mean absolute error between the predicted values and the target values.
//
// vs: the predicted values.
// tgts: the target values.
// Returns the calculated mean absolute error.
func (emae) e(vs, tgts []float64) float64 {
	var sum float64 // Initialize the sum to 0
	for i, v := range vs {
		sum += math.Abs(v - tgts[i])
	} // Return the sum
	return sum / float64(len(vs))
}

// ebce represents the binary cross entropy function.
type ebce struct{}

// e calculates the binary cross entropy between the predicted values and the target values.
//
// vs: the predicted values.
// tgts: the target values.
// Returns the calculated binary cross entropy.
func (ebce) e(vs, tgts []float64) float64 {
	var sum float64 // Initialize the sum to 0
	for i, v := range vs {
		sum += -(tgts[i]*math.Log(v) + (1-tgts[i])*math.Log(1-v))
	} // Return the sum
	return sum / float64(len(vs))
}

// ecce represents the categorical cross entropy function.
type ecce struct{}

// e calculates the categorical cross entropy between the predicted values and the target values.
//
// vs: the predicted values.
// tgts: the target values.
// Returns the calculated categorical cross entropy.
func (ecce) e(vs, tgts []float64) float64 {
	var sum float64 // Initialize the sum to 0
	for i, v := range vs {
		sum += -(tgts[i] * math.Log(v))
	} // Return the sum
	return sum / float64(len(vs))
}
