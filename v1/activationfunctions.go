package jasper

import (
	"math"
	"math/rand"
)

// type NeuralFunction func(v float64) float64
type NeuralFunction func(v float64) float64

type FunctionName int
type Functions struct {
	Activation NeuralFunction
	Derivative NeuralFunction
}

const (
	Sigmoid FunctionName = iota
	Relu
	Tanh
)

// FunctionList is a map that associates each FunctionName with its corresponding
// Functions struct. The Functions struct contains the activation and derivative
// functions for each function name.
var FunctionList map[FunctionName]Functions

// init initializes the FunctionList map. It is a special function in Go that
// runs before the main function.
func init() {
	// Initialize the FunctionList map with the available function names and their
	// corresponding activation and derivative functions.
	FunctionList = map[FunctionName]Functions{
		// Sigmoid function
		Sigmoid: {
			// Activation function for the sigmoid function
			Activation: sigmoid,
			// Derivative function for the sigmoid function
			Derivative: sigmoidDerivative,
		},
		// ReLU function
		Relu: {
			// Activation function for the ReLU function
			Activation: relu,
			// Derivative function for the ReLU function
			Derivative: reluDerivative,
		},
		// Hyperbolic Tangent function
		Tanh: {
			// Activation function for the tanh function
			Activation: tanh,
			// Derivative function for the tanh function
			Derivative: tanhDerivative,
		},
	}
}

// ApplyRandom returns a random float64 value between 0 and 1.
//
// This function is used to generate random values for initializing the weights and biases
// of the neural network during the construction of the network.
//
// Parameters:
// - v (float64): The input value. This parameter is not used in this function.
//
// Returns:
// - float64: The random float64 value between 0 and 1.
// ApplyRandom returns a random value
func ApplyRandom(v float64) float64 {
	// Generate a random float64 value between 0 and 1.
	// The rand.Float64() function generates a random float64 value in the interval [0, 1).
	return rand.Float64()
}

// Sigmoid returns the sigmoid value of the argument.
//
// The sigmoid function maps any real-valued number to a value between 0 and 1.
//
// Parameters:
// - v (float64): The input value.
//
// Returns:
// - float64: The sigmoid value of the input.
func sigmoid(v float64) float64 {
	// The sigmoid function can be expressed as 1 / (1 + exp(-x)).
	// It is commonly used in logistic regression and neural networks.
	// Here, we use the built-in math.Exp() function to calculate the exponential
	// value and subtract it from 1.
	return 1.0 / (1 + math.Exp(-v))
}

// sigmoidDerivative calculates the derivative of the sigmoid function.
//
// The derivative of the sigmoid function is given by:
// f'(x) = f(x) * (1 - f(x))
//
// Parameters:
// - v (float64): The input value for which the derivative is to be calculated.
//
// Returns:
// - float64: The derivative of the sigmoid function evaluated at v.
func sigmoidDerivative(v float64) float64 {
	// The derivative of the sigmoid function is given by:
	// f'(x) = f(x) * (1 - f(x))
	//
	// where f(x) is the sigmoid function.
	return v * (1 - v)
}

// relu calculates the Rectified Linear Unit (ReLU) activation function.
//
// The ReLU function maps any negative value to zero and leaves positive values unchanged.
//
// Parameters:
// - v (float64): The input value for which the ReLU function is to be applied.
//
// Returns:
// - float64: The output of the ReLU function.
//
// The ReLU function is defined as:
// f(x) = max(0, x)
func relu(v float64) float64 {
	// The ReLU function maps any negative value to zero and leaves positive values unchanged.
	// Here, we use the math.Max() function to calculate the maximum between 0 and the input value.
	return math.Max(0, v)
}

// reluDerivative calculates the derivative of the Rectified Linear Unit (ReLU) function.
//
// The derivative of the ReLU function is given by:
// f'(x) = 0 if x <= 0
// f'(x) = 1 if x > 0
//
// Parameters:
// - v (float64): The input value for which the derivative is to be calculated.
//
// Returns:
// - float64: The derivative of the ReLU function evaluated at v.
func reluDerivative(v float64) float64 {
	// If the input value is less than or equal to 0, return 0.
	// This is because the ReLU function sets all negative values to 0.
	if v <= 0 {
		return 0
	}

	// If the input value is greater than 0, return 1.
	// This is because the derivative of the ReLU function is 1 for all positive values.
	return 1
}

// tanh applies the Hyperbolic Tangent (tanh) function element-wise to the input tensor.
//
// The tanh function maps any real number to a value between -1 and 1, thus
// compressing the input data into a range more suitable for deep neural networks.
//
// Parameters:
// - v (float64): The input value for which the tanh function is to be applied.
//
// Returns:
// - float64: The output of the tanh function.
func tanh(v float64) float64 {
	// Apply the hyperbolic tangent function to the input value.
	// The math.Tanh() function returns the hyperbolic tangent of x.
	return math.Tanh(v)
}

// tanhDerivative calculates the derivative of the hyperbolic tangent (tanh) function.
//
// The derivative of the tanh function is given by:
// f'(x) = 1 - (tanh(x))^2
//
// Parameters:
// - v (float64): The input value for which the derivative is to be calculated.
//
// Returns:
// - float64: The derivative of the tanh function evaluated at v.
func tanhDerivative(v float64) float64 {
	// Calculate the derivative of the tanh function using the formula:
	// f'(x) = 1 - (tanh(x))^2
	// where x is the input value.
	return 1 - (v * v)
}
