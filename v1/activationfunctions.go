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

var FunctionList map[FunctionName]Functions

const (
	Sigmoid FunctionName = iota
	Relu
	Tanh
)

func init() {
	FunctionList = map[FunctionName]Functions{

		Sigmoid: {
			Activation: sigmoid,
			Derivative: dsigmoid,
		},
		Relu: {
			Activation: relu,
			Derivative: drelu,
		},
		Tanh: {
			Activation: tanh,
			Derivative: dtanh,
		},
	}

}

// ApplyRandom returns a random value
func ApplyRandom(v float64) float64 {
	return rand.Float64()
}

// Sigmoid returns the sigmoid value of the argument
// func sigmoid(v float64) float64 {
func sigmoid(v float64) float64 {
	return 1.0 / (1 + math.Exp(-v))
}

// Dsigmoid returns derivative of sigmoid function
// func dsigmoid(v float64) float64 {
func dsigmoid(v float64) float64 {
	return v * (1 - v)
}

// func relu(v float64) float64 {
func relu(v float64) float64 {
	return math.Max(0, v)
}

// func drelu(v float64) float64 {
func drelu(v float64) float64 {
	if v <= 0 {
		return 0
	}
	return 1
}

// func tanh(v float64) float64 {
func tanh(v float64) float64 {
	return math.Tanh(v)
}

// func dtanh(v float64) float64 {
func dtanh(v float64) float64 {
	return 1 - (v * v)
}
