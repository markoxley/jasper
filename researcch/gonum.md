# Gonum

This is code provided by ChatGPT

```go
package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"time"
)

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of sigmoid
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// NeuralNetwork represents a simple neural network
type NeuralNetwork struct {
	inputs        *mat.Dense
	weightsInput  *mat.Dense
	weightsHidden *mat.Dense
	biasHidden    *mat.Dense
	biasOutput    *mat.Dense
}

// NewNeuralNetwork initializes the neural network
func NewNeuralNetwork(inputNeurons, hiddenNeurons, outputNeurons int) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())

	// Randomly initialize weights and biases
	weightsInput := mat.NewDense(inputNeurons, hiddenNeurons, nil)
	weightsHidden := mat.NewDense(hiddenNeurons, outputNeurons, nil)
	biasHidden := mat.NewDense(1, hiddenNeurons, nil)
	biasOutput := mat.NewDense(1, outputNeurons, nil)

	for i := 0; i < inputNeurons; i++ {
		for j := 0; j < hiddenNeurons; j++ {
			weightsInput.Set(i, j, rand.Float64())
		}
	}
	for i := 0; i < hiddenNeurons; i++ {
		for j := 0; j < outputNeurons; j++ {
			weightsHidden.Set(i, j, rand.Float64())
		}
	}
	for i := 0; i < hiddenNeurons; i++ {
		biasHidden.Set(0, i, rand.Float64())
	}
	for i := 0; i < outputNeurons; i++ {
		biasOutput.Set(0, i, rand.Float64())
	}

	return &NeuralNetwork{
		weightsInput:  weightsInput,
		weightsHidden: weightsHidden,
		biasHidden:    biasHidden,
		biasOutput:    biasOutput,
	}
}

// Feedforward performs a forward pass through the network
func (nn *NeuralNetwork) Feedforward(inputs []float64) []float64 {
	inputMat := mat.NewDense(1, len(inputs), inputs)

	// Calculate hidden layer activations
	hiddenInput := new(mat.Dense)
	hiddenInput.Mul(inputMat, nn.weightsInput)
	hiddenInput.Add(hiddenInput, nn.biasHidden)

	hiddenOutput := mat.NewDense(hiddenInput.RawMatrix().Rows, hiddenInput.RawMatrix().Cols, nil)
	applySigmoid(hiddenInput, hiddenOutput)

	// Calculate output layer activations
	finalInput := new(mat.Dense)
	finalInput.Mul(hiddenOutput, nn.weightsHidden)
	finalInput.Add(finalInput, nn.biasOutput)

	finalOutput := mat.NewDense(finalInput.RawMatrix().Rows, finalInput.RawMatrix().Cols, nil)
	applySigmoid(finalInput, finalOutput)

	return finalOutput.RawRowView(0)
}

// applySigmoid applies the sigmoid function to all elements in the matrix
func applySigmoid(input, output *mat.Dense) {
	r, c := input.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			output.Set(i, j, sigmoid(input.At(i, j)))
		}
	}
}

func main() {
	// Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
	nn := NewNeuralNetwork(2, 2, 1)

	// Example inputs
	inputs := []float64{0.5, 0.9}

	// Perform a forward pass
	output := nn.Feedforward(inputs)

	fmt.Printf("Output: %v\n", output)
}

```
