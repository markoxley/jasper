package jasper

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Network represents a neural network.
type Network struct {
	// topology is the configuration of the network, a slice of uint32 that
	// represents the number of neurons in each layer.
	topology []uint32

	// weightMatrices is a slice of weight matrices, each matrix is a connection
	// between two layers.
	weightMatrices []*Matrix

	// valueMatrices is a slice of matrices that represent the output values of
	// each layer.
	valueMatrices []*Matrix

	// biasMatrices is a slice of bias matrices, each matrix is a bias for each
	// layer.
	biasMatrices []*Matrix

	// learningRate is a float64 that represents the learning rate of the network.
	learningRate float64

	// activation is the activation function used in the network.
	activation ActivationFunction

	// solver is the solver for the activation function.
	solver ActivationSolver

	// output is the output activation function of the network.
	output ActivationFunction

	// outputSolver is the solver for the output activation function.
	outputSolver ActivationSolver

	// errFunc is the error function used in the network.
	errFunc ErrorFunction

	// errorSolver is the solver for the error function.
	errorSolver ErrorSolver

	// debug is a boolean that indicates if the network is in debug mode.
	debug bool

	// sm is a boolean that indicates if the network should use soft max.
	sm bool
}

// getRandom generates a random float64 using the math/rand package.
//
// The parameter is unused and is only included to maintain the same function
// signature as getRandomFloats.
//
// Returns:
// - A random float64.
func getRandom(unused float64) float64 {
	// Generate a random float64 using the math/rand package.
	// The random float64 is between 0 and 1.
	return rand.Float64()
}

// getRandomFloats generates an array of random floats.
//
// Parameters:
// - sz: The size of the array to generate.
//
// Returns:
// - An array of random floats with the length specified by the parameter 'sz'.
func getRandomFloats(sz int) []float64 {
	// Create a slice of the specified size.
	r := make([]float64, sz)

	// Iterate over each element of the slice.
	for i := range r {
		// Generate a random float using the ApplyRandom function and assign it to the current element of the slice.
		r[i] = getRandom(0)
	}

	// Return the generated slice of random floats.
	return r
}

// softMax calculates the softmax function on a given Matrix.
//
// The softmax function is used to normalize a set of values into a probability distribution.
// It takes a Matrix as input and returns a new Matrix with the same dimensions.
//
// Parameters:
// - vs: A pointer to a Matrix representing the input values.
//
// Returns:
// - A pointer to a new Matrix with the same dimensions as the input Matrix, containing the softmax values.
func softMax(vs *Matrix) *Matrix {
	// Calculate the total sum of the exponentials of the input values.
	// This is used to normalize the output values.
	var total float64

	// Create an output slice to hold the result of applying the softmax function.
	output := make([]float64, len(vs.Values()))

	// Iterate over the input slice and calculate the exponential of each value.
	// Add each value to the total sum.
	for i, v := range vs.values {
		output[i] = math.Exp(v)
		total += v
	}

	// Iterate over the output slice and divide each value by the total sum.
	// This normalizes the output values to be between 0 and 1.
	for i := range vs.values {
		output[i] /= total
	}

	// Return the output slice.
	return NewMatrixFromSlice(output)
}

// New creates a new instance of the Network struct.
//
// Parameters:
// - c: A pointer to the NetworkConfiguration struct that contains the configuration settings for the network.
//
// Returns:
// - A pointer to the newly created Network struct and an error if any.
func New(c *NetworkConfiguration) (*Network, error) {
	// Create a new instance of the Network struct using the configuration settings.
	s := Network{
		topology:     c.Topology,                           // Set the topology of the network.
		learningRate: c.LearningRate,                       // Set the learning rate of the network.
		activation:   c.Activation,                         // Set the function name of the network.
		solver:       GetActivationFunctions(c.Activation), // Set the activation function of the network.
		output:       c.Output,
		outputSolver: GetActivationFunctions(c.Output),
		errFunc:      c.Error,
		errorSolver:  GetErrorFunction(c.Error),
		debug:        !c.Quiet, // Set the debug mode of the network.
		sm:           c.SoftMax,
	}

	// Iterate over each layer of the network.
	for i := 0; i < len(s.topology)-1; i++ {
		// Create a new weight matrix for the current layer.
		wm := NewMatrix(s.topology[i+1], s.topology[i])                          // Set the dimensions of the weight matrix.
		s.weightMatrices = append(s.weightMatrices, wm.ApplyFunction(getRandom)) // Apply a random function to each element of the weight matrix.

		// Create a new bias matrix for the current layer.
		bm := NewMatrix(s.topology[i+1], 1)                                  // Set the dimensions of the bias matrix.
		s.biasMatrices = append(s.biasMatrices, bm.ApplyFunction(getRandom)) // Apply a random function to each element of the bias matrix.
	}

	// Create a slice to store the value matrices for each layer.
	s.valueMatrices = make([]*Matrix, len(s.topology))

	// Return the newly created Network struct.
	return &s, nil
}

// feedForward performs a feed-forward operation on the network.
//
// Parameters:
// - input: A slice of floats representing the input values.
//
// Returns:
// - An error if the input size is incorrect.
func (n *Network) feedForward(input []float64) error {
	// Check if the input size is correct.
	if len(input) != int(n.topology[0]) {
		return errors.New("incorrect input size")
	}

	// Create a new matrix to hold the input values.
	values := NewMatrix(uint32(len(input)), 1)

	// Populate the input values into the matrix.
	for i, in := range input {
		values.Set(uint32(i), 0, in)
	}

	var err error

	// Feed forward to each layer.
	for i, w := range n.weightMatrices {
		// Set the current layer's values to the input values.
		n.valueMatrices[i] = values

		// Multiply the input values with the weight matrix.
		values, err = values.Multiply(w)
		if err != nil {
			return fmt.Errorf("feed forward error: %v", err)
		}

		// Add the bias values to the current layer's values.
		values, err = values.Add(n.biasMatrices[i])
		if err != nil {
			return fmt.Errorf("feed forward error: %v", err)
		}

		// Apply the activation function to the current layer's values.
		if i < len(n.weightMatrices)-1 {
			values = values.ApplyFunction(n.solver.F)
		} else {
			values = values.ApplyFunction(n.outputSolver.F)
		}
	}

	// Set the output values of the network to the final layer's values.
	n.valueMatrices[len(n.weightMatrices)] = values

	if n.sm {
		n.valueMatrices[len(n.weightMatrices)] = softMax(values)
	} else {
		n.valueMatrices[len(n.weightMatrices)] = values
	}
	// Return nil if there are no errors.
	return nil
}

// backPropagate performs the back propagation operation on the network.
//
// Parameters:
// - tgtOut: A slice of floats representing the target output values.
//
// Returns:
// - An error if the target output size is incorrect.
func (n *Network) backPropagate(tgtOut []float64) error {
	// Check if the target output size is correct.
	if len(tgtOut) != int(n.topology[len(n.topology)-1]) {
		return errors.New("output is incorrect size")
	}

	// Create a new matrix to hold the target output values.
	errMtx := NewMatrix(uint32(len(tgtOut)), 1)

	// Populate the target output values into the matrix.
	errMtx.SetValues(tgtOut)

	// Calculate the error matrix.
	errMtx, err := errMtx.Add(n.valueMatrices[len(n.valueMatrices)-1].Negative())
	if err != nil {
		return fmt.Errorf("back propagation error: %v", err)
	}

	// Iterate through the layers from the last layer to the first layer.
	for i := len(n.weightMatrices) - 1; i >= 0; i-- {
		// Calculate the error at the current layer.
		prevErrors, err := errMtx.Multiply(n.weightMatrices[i].Transpose())
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}

		// Apply the derivative of the activation function to the output values of the current layer.
		dOutputs := n.valueMatrices[i+1].ApplyFunction(n.solver.Df)

		// Calculate the gradients of the error with respect to the weights and biases.
		gradients, err := errMtx.MultiplyElements(dOutputs)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}
		gradients = gradients.MultiplyScalar(n.learningRate)

		// Calculate the weight gradients.
		weightGradients, err := n.valueMatrices[i].Transpose().Multiply(gradients)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}

		// Update the weight matrices.
		n.weightMatrices[i], err = n.weightMatrices[i].Add(weightGradients)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}

		// Update the bias matrices.
		n.biasMatrices[i], err = n.biasMatrices[i].Add(gradients)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}

		// Update the error matrix for the next iteration.
		errMtx = prevErrors
	}

	return nil
}

// getPrediction returns the values of the output layer of the network.
//
// This function does not take any parameters.
// It returns a slice of floats representing the output values of the network.
func (n *Network) getPrediction() []float64 {
	// The output values of the network are stored in the last value matrix.
	// We return the values of this matrix.
	// The Values() function returns a slice of floats representing the values of the matrix.
	return n.valueMatrices[len(n.valueMatrices)-1].Values()
}

// Train trains the network using the training data.
//
// This function takes a TrainingData object as a parameter and returns the average
// error and an error object.
//
// The function iterates over the training data for the specified number of iterations.
// During each iteration, it feeds the input data through the network and backpropagates
// the error to update the network's weights and biases.
// After each iteration, it checks if the network's error is within the specified tolerance.
// If it is, the training process is terminated early.
// The function returns the average error and a nil error object if the training is successful.
func (n *Network) Train(td *TrainingData) (float64, error) {
	// Initialize the training process and print debug information if debug mode is enabled
	var start time.Time
	if n.debug {
		fmt.Println("initialising training...")
		fmt.Printf("\t%v input neurons\n", n.topology[0])
		fmt.Printf("\t%v hidden layers\n", len(n.topology)-2)
		for i := 1; i < len(n.topology)-1; i++ {
			fmt.Printf("\t\t%v neurons\n", n.topology[i])
		}
		fmt.Printf("\t%v output neurons\n", n.topology[len(n.topology)-1])
		totalNeuronCount := 0
		totalSynapsCount := 0
		last := 0
		for _, nc := range n.topology {
			totalNeuronCount += int(nc)
			totalSynapsCount += (int(nc) * last)
			last = int(nc)
		}
		fmt.Printf("\t%v total neuron count\n", totalNeuronCount)
		fmt.Printf("\t%v total synapse count\n", totalSynapsCount)
		fmt.Println("\npreparing data")
	}
	td.Prepare()
	if n.debug {
		fmt.Printf("\t%v rows of training data\n", td.TrainingCount())
		fmt.Printf("\t%v rows of testing data\n", td.TestCount())
	}
	var errSum float64
	if n.debug {
		start = time.Now()
		fmt.Printf("\ntraining commencing at %v\n", start)
	}
	iterCount := 0 // Keep track of the number of iterations
	for i := 0; i < int(td.Iterations); i++ {

		// Print a dot for each 1000 iterations and a new line for each 80,000 iterations
		if n.debug {
			iterCount++
			if i%1000 == 0 {
				if i > 0 && i%10000 == 0 {
					fmt.Printf(" %v\n", i)
				}
				fmt.Print(".")
			}
		}
		// Iterate over the training data and feed it through the network
		for {
			row := td.NextRow()
			if row == nil {
				break
			}
			if err := n.feedForward(row.Input); err != nil {
				return 0, fmt.Errorf("training error: %v", err)
			}
			if err := n.backPropagate(row.Ouput); err != nil {
				return 0, fmt.Errorf("training error: %v", err)
			}
		}
		errSum = 0
		errorWithinTolerence := true
		testCount := 0
		// Calculate the average error for the testing data
		for _, errCheck := range td.TestData() {
			testCount++
			answer, err := n.Predict(errCheck.Input)
			if err != nil {
				return 0, fmt.Errorf("error testing error value: %v", err)
			}
			v := n.errorSolver.Calculate(errCheck.Ouput, answer)
			if v > td.TargetError {
				errorWithinTolerence = false
			}
			errSum += v
			// var v float64
			// for i, a := range answer {
			// 	v += math.Pow(errCheck.Ouput[i]-a, 2)
			// }
			// v /= float64(len(answer))
			// // Check if the error is within the specified tolerance
			// if math.Sqrt(v) > td.TargetError {
			// 	errorWithinTolerence = false
			// 	break
			// }
			// errSum += v
		}
		// errSum = math.Sqrt(errSum / float64(len(td.TestData())))
		errSum /= float64(testCount)
		// Check if the error is within the specified tolerance
		if errorWithinTolerence && errSum <= td.TargetError {
			if n.debug {
				fmt.Print("\nterminating early. Within tolerance.")
			}
			break
		}
	}
	// Print the training completion time and the number of iterations if debug mode is enabled
	if n.debug {
		stop := time.Now()
		fmt.Printf("\ntraining complete at %v\n", stop)
		fmt.Printf("training took %v minutes\n", stop.Sub(start).Minutes())
		fmt.Printf("\t%v iterations run\n", iterCount)
		fmt.Printf("\terror margin is %0.5f\n", errSum)
	}
	// Return the average error and a nil error object if the training is successful
	return errSum, nil
}

// Predict uses the network to predict the output given an input.
// It performs a feed-forward operation on the network and returns the predicted output.
//
// Parameters:
// - input: A slice of floats representing the input values.
//
// Returns:
// - A slice of floats representing the predicted output values.
// - An error if there is an error during the prediction.
func (n *Network) Predict(input []float64) ([]float64, error) {
	// Perform a feed-forward operation on the network.
	err := n.feedForward(input)
	if err != nil {
		// Return an error if there is an error during the feed-forward operation.
		return nil, fmt.Errorf("prediction error: %v", err)
	}
	// Return the predicted output values.
	return n.getPrediction(), nil
}

// SetDebug sets the debug mode of the network.
//
// The debug mode determines whether debug information is printed during the training process.
//
// Parameters:
// - v: A boolean value indicating whether the debug mode is enabled (true) or disabled (false).
func (n *Network) SetDebug(v bool) {
	// Set the debug mode of the network to the specified value.
	n.debug = v
}

// Debug returns the debug mode of the network.
//
// The debug mode determines whether debug information is printed during the training process.
//
// Returns:
// - A boolean value indicating whether the debug mode is enabled (true) or disabled (false).
func (n *Network) Debug() bool {
	// Return the debug mode of the network.
	return n.debug
}

// MarshalJSON marshals the Network object into a JSON byte slice.
//
// Parameters:
// - None
//
// Returns:
// - A JSON byte slice representing the Network object.
// - An error if there is an error during the marshaling process.
func (n *Network) MarshalJSON() ([]byte, error) {

	res := struct {
		Topology       []uint32  `json:"t"`
		WeightMatrices []*Matrix `json:"w"`
		BiasMatrices   []*Matrix `json:"b"`
		LearningRate   float64   `json:"k"`
		Activation     int       `json:"a"`
		Output         int       `json:"o"`
		ErrFunc        int       `json:"e"`
		Debug          bool      `json:"d"`
		SM             bool      `json:"s"`
	}{
		Topology:       n.topology,
		WeightMatrices: n.weightMatrices,
		BiasMatrices:   n.biasMatrices,
		LearningRate:   n.learningRate,
		Activation:     int(n.activation),
		Output:         int(n.output),
		ErrFunc:        int(n.errFunc),
		Debug:          n.debug,
		SM:             n.sm,
	}

	return json.Marshal(&res)
}

// UnmarshalJSON unmarshals the JSON byte slice into the Network object.
//
// Parameters:
// - body (byte slice): The JSON byte slice to be unmarshaled.
//
// Returns:
// - err (error): An error if there is an error during the unmarshaling process.
func (n *Network) UnmarshalJSON(body []byte) (err error) {
	data := struct {
		Topology       []uint32  `json:"t"`
		WeightMatrices []*Matrix `json:"w"`
		BiasMatrices   []*Matrix `json:"b"`
		LearningRate   float64   `json:"k"`
		Activation     int       `json:"a"`
		Output         int       `json:"o"`
		ErrFunc        int       `json:"e"`
		Debug          bool      `json:"d"`
		SM             bool      `json:"s"`
	}{}
	if err := json.Unmarshal(body, &data); err != nil {
		return err
	}
	n.topology = data.Topology
	n.weightMatrices = data.WeightMatrices
	n.biasMatrices = data.BiasMatrices
	n.learningRate = data.LearningRate
	n.activation = ActivationFunction(data.Activation)
	n.output = ActivationFunction(data.Output)
	n.errFunc = ErrorFunction(data.ErrFunc)
	n.debug = data.Debug
	n.sm = data.SM
	n.solver = GetActivationFunctions(n.activation)
	n.outputSolver = GetActivationFunctions(n.output)
	n.errorSolver = GetErrorFunction(n.errFunc)
	n.valueMatrices = make([]*Matrix, len(n.topology))
	return nil
}
