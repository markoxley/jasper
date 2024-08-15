// activationfunctions.go - Activation functions used in the neural network.
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

// ActivationFunction is an enumeration of the different activation functions
// that can be used in a neural network.
//
// The constants that can be used are Sigmoid, Relu, Tanh, LeakyRelu, Softplus,
// Elu, Gelu, Swish, and Linear.
type ActivationFunction int

// neuralFunction is a function that takes a float64 and returns a float64.
// It is used by the activation functions in the network.
type neuralFunction func(v float64) float64

const (
	// Sigmoid is the sigmoid activation function.
	Sigmoid ActivationFunction = iota
	// Relu is the rectified linear unit activation function.
	Relu
	// Tanh is the hyperbolic tangent activation function.
	Tanh
	// LeakyRelu is the leaky rectified linear unit activation function.
	LeakyRelu
	// Softplus is the softplus activation function.
	Softplus
	// Swish is the swish activation function.
	Swish
	// ELU is the exponential linear unit activation function.
	ELU
	// GELU is the Gaussian exponential linear unit activation function.
	GELU
	// Linear is the linear activation function.
	Linear
)

// getActivationFunctions returns an instance of the ActivationSolver interface for the given ActivationFunction.
//
// Parameters:
// - name: The name of the activation function.
//
// Returns:
// - ActivationSolver: An instance of the ActivationSolver interface.
func getActivationFunctions(name ActivationFunction) activationSolver {
	switch name {
	case Sigmoid:
		return fsigmoid{}
	case Relu:
		return frelu{}
	case Tanh:
		return ftanh{}
	case LeakyRelu:
		return fleakyrelu{}
	case Softplus:
		return fsoftlus{}
	case Swish:
		return fswish{}
	case ELU:
		return felu{}
	case GELU:
		return fgelu{}
	case Linear:
		return flinear{}
	}
	return nil
}

// activationSolver is an interface used to abstract away the underlying
// implementation details of activation functions. It is used to provide a
// consistent interface for activation functions.
//
// The interface is comprised of two methods:
// - F(v float64): Used to compute the output of the activation function.
// - Df(v float64): Used to compute the derivative of the activation function.
type activationSolver interface {
	// f computes the output of the activation function given the input v.
	f(v float64) float64
	// df computes the derivative of the activation function given the input v.
	df(v float64) float64
}

// fsigmoid is an implementation of the sigmoid activation function.
//
// It implements the ActivationSolver interface, which is used to abstract away
// the underlying implementation details of activation functions.
type fsigmoid struct{}

// f computes the output of the sigmoid activation function.
//
// Parameters:
// - v (float64): The input value.
//
// Returns:
// - float64: The output of the sigmoid activation function.
func (fsigmoid) f(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

// df computes the derivative of the sigmoid activation function given the input v.
//
// Parameters:
// - v (float64): The input value.
//
// Returns:
// - float64: The derivative of the sigmoid activation function.
func (fsigmoid) df(v float64) float64 {
	return v * (1 - v)
}

// frelu is an implementation of the ReLU (Rectified Linear Unit) activation
// function.
//
// It implements the ActivationSolver interface, which is used to abstract away
// the underlying implementation details of activation functions.
type frelu struct{}

// f computes the output of the ReLU activation function.
//
// Parameters:
// - v (float64): The input value to the ReLU activation function.
//
// Returns:
// - float64: The output of the ReLU activation function, which is the maximum of 0 and the input value.
func (frelu) f(v float64) float64 {
	return math.Max(0, v)
}

// df computes the derivative of the ReLU activation function given the input v.
//
// Parameters:
// - v (float64): The input value.
//
// Returns:
// - float64: The derivative of the ReLU activation function.
func (frelu) df(v float64) float64 {
	if v > 0 {
		return 1
	}
	return 0
}

// ftanh is an implementation of the hyperbolic tangent activation function.
//
// It implements the ActivationSolver interface, which is used to abstract away
// the underlying implementation details of activation functions.
type ftanh struct{}

// f computes the output of the hyperbolic tangent activation function.
//
// Parameters:
// - v (float64): The input value to the hyperbolic tangent activation function.
//
// Returns:
// - float64: The output of the hyperbolic tangent activation function.
func (ftanh) f(v float64) float64 {
	return math.Tanh(v)
}

// df computes the derivative of the hyperbolic tangent activation function.
//
// Parameters:
// - v (float64): The input value.
//
// Returns:
// - float64: The derivative of the hyperbolic tangent activation function.
func (ftanh) df(v float64) float64 {
	return 1 - (v * v)
}

// fleakyrelu is an implementation of the leaky ReLU activation function.
//
// It implements the ActivationSolver interface, which is used to abstract away
// the underlying implementation details of activation functions.
type fleakyrelu struct{}

// f computes the output of the leaky ReLU activation function.
//
// Parameters:
// - v (float64): The input value to the leaky ReLU activation function.
//
// Returns:
// - float64: The output of the leaky ReLU activation function.
func (fleakyrelu) f(v float64) float64 {
	if v > 0 {
		return v
	}
	return 0.01 * v
}

// df computes the derivative of the Leaky ReLU activation function.
//
// Parameters:
// - v (float64): The input value.
//
// Returns:
// - float64: The derivative of the Leaky ReLU activation function. If the input is greater than 0, it returns 1. Otherwise, it returns 0.01.
func (fleakyrelu) df(v float64) float64 {
	if v > 0 {
		return 1
	}
	return 0.01
}

// flinear is a struct representing the linear activation function.
//
// Linear stands for the identity function, where the output is equal to the input.
type flinear struct{}

// f computes the output of the linear activation function.
//
// Parameters:
// - v (float64): The input value to the linear activation function.
//
// Returns:
// - float64: The output of the linear activation function, which is the same as the input value.
func (flinear) f(v float64) float64 {
	return v
}

// df computes the derivative of the linear activation function.
//
// Parameters:
// - v (float64): The input value.
//
// Returns:
// - float64: The derivative of the linear activation function.
func (flinear) df(v float64) float64 {
	return 1
}

// fswish is a struct that represents the Swish activation function.
//
// The Swish activation function is also known as the SiLU (Sigmoid-weighted Linear Unit) function.
// It is defined as:
//
//     f(x) = x / (1 + exp(-x))
//
type fswish struct{}

// f calculates the output of the fswish function.
//
// Parameters:
// - v (float64): The input value to the fswish function.
//
// Returns:
// - float64: The output of the fswish function.
func (fswish) f(v float64) float64 {
	return v / (1 + math.Exp(-v))
}

// df computes the derivative of the Swish activation function.
//
// Parameters:
// - v (float64): The input value.
//
// Returns:
// - float64: The derivative of the Swish activation function.
func (fswish) df(v float64) float64 {
	return v / (1 + math.Exp(-v)) * (1 - v/(1+math.Exp(-v)))
}

// felu is a struct representing the Exponential Linear Unit (ELU) activation function.
//
// ELU is an activation function that is similar to ReLU but has a smoother gradient.
// It is defined as f(x) = x if x >= 0, and f(x) = a * (exp(x) - 1) if x < 0.
type felu struct{}

// f computes the output of the ELU activation function.
//
// Parameters:
// - v (float64): The input value to the ELU activation function.
//
// Returns:
// - float64: The output of the ELU activation function.
func (felu) f(v float64) float64 {
	if v > 0 {
		return v
	}
	return math.Exp(v) - 1
}

// df computes the derivative of the ELU activation function.
//
// Parameters:
// - v (float64): The input value.
//
// Returns:
// - float64: The derivative of the ELU activation function.
func (felu) df(v float64) float64 {
	if v > 0 {
		return 1
	}
	return math.Exp(v)
}

// fgelu is a struct representing the GELU activation function.
//
// GELU stands for Gaussian Error Linear Unit.
type fgelu struct{}

// f calculates the output of the GELU activation function.
//
// Parameters:
// - v (float64): The input value to the GELU activation function.
//
// Returns:
// - float64: The output of the GELU activation function.
func (fgelu) f(v float64) float64 {
	return 0.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*math.Pow(v, 3))))
}

// df computes the derivative of the GELU activation function.
//
// Parameters:
// - v (float64): The input value.
//
// Returns:
// - float64: The derivative of the GELU activation function.
func (fgelu) df(v float64) float64 {
	return 0.5*(1+math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*math.Pow(v, 3)))) + 0.5*math.Pow(math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*math.Pow(v, 3))), 2)
}

// fsoftlus is an implementation of the Softplus activation function.
//
// The Softplus activation function is a continuous, differentiable
// approximation of the ReLU activation function. It is defined as
// f(x) = log(1 + exp(x)).
type fsoftlus struct{}

// f calculates the output of the Softplus activation function.
//
// Parameters:
// - v (float64): The input value to the Softplus activation function.
//
// Returns:
// - float64: The output of the Softplus activation function.
func (fsoftlus) f(v float64) float64 {
	return math.Log(1 + math.Exp(v))
}

// df computes the derivative of the Softplus activation function.
//
// Parameters:
// - v (float64): The input value.
//
// Returns:
// - float64: The derivative of the Softplus activation function.
func (fsoftlus) df(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}
