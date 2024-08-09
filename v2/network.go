package jasper

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type SaveData struct {
	Topology       []uint32          `json:"t"`
	WeightMatrices []*MatrixSaveData `json:"w"`
	BiasMatrices   []*MatrixSaveData `json:"b"`
	LearningRate   float64           `json:"l"`
	Functions      uint32            `json:"f"`
}
type Network struct {
	topology      []uint32
	inputWeights  *mat.Dense
	hiddenWeights []*mat.Dense
	biasHidden    []*mat.Dense
	biasOutput    *mat.Dense
	learningRate  float64
	functionName  FunctionName
	activation    NeuralFunction
	derivative    NeuralFunction
	debug         bool
	Result        []float64
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
		r[i] = ApplyRandom(0)
	}

	// Return the generated slice of random floats.
	return r
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
	n := Network{
		topology:     c.Topology,                           // Set the topology of the network.
		learningRate: c.LearningRate,                       // Set the learning rate of the network.
		functionName: c.Functions,                          // Set the function name of the network.
		activation:   FunctionList[c.Functions].Activation, // Set the activation function of the network.
		derivative:   FunctionList[c.Functions].Derivative, // Set the derivative function of the network.
		debug:        !c.Quiet,                             // Set the debug mode of the network.
	}
	inputNeurons := int(c.Topology[0])
	hiddenLayers := len(c.Topology) - 2
	outputNeurons := int(c.Topology[2])

	n.inputWeights = mat.NewDense(inputNeurons, int(n.topology[1]), nil)
	for i := 0; i < inputNeurons; i++ {
		for j := 0; j < int(n.topology[1]); j++ {
			n.inputWeights.Set(i, j, rand.Float64())
		}
	}

	n.hiddenWeights = make([]*mat.Dense, hiddenLayers)
	for i := range n.hiddenWeights {
		n.hiddenWeights[i] = mat.NewDense(int(n.topology[i+1]), int(n.topology[i+2]), nil)
		for j := 0; j < int(n.topology[i+1]); j++ {
			for k := 0; k < int(n.topology[i+2]); k++ {
				n.hiddenWeights[i].Set(j, k, rand.Float64())
			}
		}
	}

	n.biasHidden = make([]*mat.Dense, hiddenLayers)
	for i := range n.biasHidden {
		n.biasHidden[i] = mat.NewDense(1, int(n.topology[i+2]), nil)
		for j := 0; j < int(n.topology[i+2]); j++ {
			n.biasHidden[i].Set(0, j, rand.Float64())
		}
	}

	n.biasOutput = mat.NewDense(1, outputNeurons, nil)
	for i := 0; i < outputNeurons; i++ {
		n.biasOutput.Set(0, i, rand.Float64())
	}

	// Return the newly created Network struct.
	return &n, nil
}

func (n *Network) applyFunction(f NeuralFunction, input, output *mat.Dense) {
	r, c := input.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			output.Set(i, j, f(input.At(i, j)))
		}
	}
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

	data := mat.NewDense(1, len(input), input)

	for i := 0; i < len(n.hiddenWeights)+1; i++ {
		var wgts *mat.Dense
		var bias *mat.Dense
		if i < 1 {
			wgts = n.inputWeights
		} else {
			wgts = n.hiddenWeights[i-1]
		}
		if i == len(n.hiddenWeights)-1 {
			bias = n.biasOutput
		} else {
			bias = n.biasHidden[i]
		}
		hiddenInput := new(mat.Dense)
		hiddenInput.Mul(data, wgts)
		hiddenInput.Add(hiddenInput, bias)
		n.applyFunction(n.activation, hiddenInput, data)
	}

	result := mat.NewDense(data.RawMatrix().Rows, data.RawMatrix().Cols, nil)
	n.applyFunction(n.activation, data, result)

	n.Result = result.RawRowView(0)
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

	errVec := mat.NewVecDense(len(n.Result), n.Result)
	resVec := mat.NewVecDense(len(tgtOut), tgtOut)

	// Calculate the Mean Squared Error
	mse := mat.NewVecDense(1, nil)
	mse.SubVec(errVec, resVec)
	mse.
		mse.PowVec(mse, 2)
	mse.Mean(mse)

	// Iterate through the layers from the last layer to the first layer.
	// for i := len(n.weightMatrices) - 1; i >= 0; i-- {
	// 	// Calculate the error at the current layer.
	// 	prevErrors, err := errMtx.Multiply(n.weightMatrices[i].Transpose())
	// 	if err != nil {
	// 		return fmt.Errorf("back propagation error: %v", err)
	// 	}

	// 	// Apply the derivative of the activation function to the output values of the current layer.
	// 	dOutputs := n.valueMatrices[i+1].ApplyFunction(n.derivative)

	// 	// Calculate the gradients of the error with respect to the weights and biases.
	// 	gradients, err := errMtx.MultiplyElements(dOutputs)
	// 	if err != nil {
	// 		return fmt.Errorf("back propagation error: %v", err)
	// 	}
	// 	gradients = gradients.MultiplyScalar(n.learningRate)

	// 	// Calculate the weight gradients.
	// 	weightGradients, err := n.valueMatrices[i].Transpose().Multiply(gradients)
	// 	if err != nil {
	// 		return fmt.Errorf("back propagation error: %v", err)
	// 	}

	// 	// Update the weight matrices.
	// 	n.weightMatrices[i], err = n.weightMatrices[i].Add(weightGradients)
	// 	if err != nil {
	// 		return fmt.Errorf("back propagation error: %v", err)
	// 	}

	// 	// Update the bias matrices.
	// 	n.biasMatrices[i], err = n.biasMatrices[i].Add(gradients)
	// 	if err != nil {
	// 		return fmt.Errorf("back propagation error: %v", err)
	// 	}

	// 	// Update the error matrix for the next iteration.
	// 	errMtx = prevErrors
	// }

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
	return n.Result
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
				if i > 0 && i%80_000 == 0 {
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
		// Calculate the average error for the testing data
		for _, errCheck := range td.TestData() {
			answer, err := n.Predict(errCheck.Input)
			if err != nil {
				return 0, fmt.Errorf("error testing error value: %v", err)
			}
			var v float64
			for i, a := range answer {
				v += math.Pow(errCheck.Ouput[i]-a, 2)
			}
			v /= float64(len(answer))
			// Check if the error is within the specified tolerance
			if math.Sqrt(v) > td.TargetError {
				errorWithinTolerence = false
				break
			}
			errSum += v
		}
		errSum = math.Sqrt(errSum / float64(len(td.TestData())))
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
		fmt.Printf("\terror margin is %v\n", errSum)
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

// // ToSaveData converts the network to a SaveData object, which can be used to save the network's state.
// //
// // It returns a pointer to a SaveData object.
// func (n *Network) ToSaveData() *SaveData {
// 	// Create a new SaveData object.
// 	sd := SaveData{
// 		// Set the topology, learning rate, and function name.
// 		Topology:     n.topology,
// 		LearningRate: n.learningRate,
// 		Functions:    uint32(n.functionName),
// 		// Create slices to hold the weight and bias matrices' save data.
// 		WeightMatrices: make([]*MatrixSaveData, len(n.weightMatrices)),
// 		BiasMatrices:   make([]*MatrixSaveData, len(n.biasMatrices)),
// 	}
// 	// Convert each weight matrix to save data and add it to the save data object.
// 	for i, wm := range n.weightMatrices {
// 		sd.WeightMatrices[i] = wm.ToSaveData()
// 	}
// 	// Convert each bias matrix to save data and add it to the save data object.
// 	for i, bm := range n.biasMatrices {
// 		sd.BiasMatrices[i] = bm.ToSaveData()
// 	}
// 	// Return the save data object.
// 	return &sd
// }

// // ToJson converts the network to its JSON representation.
// //
// // It returns the JSON representation as a byte slice and an error if there is an error during the conversion.
// func (n *Network) ToJson() ([]byte, error) {
// 	// Convert the network to a SaveData object.
// 	saveData := n.ToSaveData()
// 	// Convert the SaveData object to its JSON representation.
// 	// The json.Marshal function is used to convert the SaveData object to its JSON representation.
// 	// The returned byte slice contains the JSON representation of the SaveData object.
// 	// The error is returned if there is an error during the conversion.
// 	return json.Marshal(saveData)
// }

// // Write writes the network's JSON representation to the provided writer.
// // It returns an error if there is an error during the conversion or writing process.
// func (n *Network) Write(w io.Writer) error {
// 	// Convert the network to its JSON representation.
// 	j, err := n.ToJson()
// 	if err != nil {
// 		return fmt.Errorf("network write error: %v", err)
// 	}

// 	// Write the JSON representation to the writer.
// 	// The Write method of the writer is used to write the JSON representation.
// 	// The number of bytes written is returned.
// 	// If there is an error during the writing process, an error is returned.
// 	c, err := w.Write(j)
// 	if err != nil {
// 		return fmt.Errorf("network write error: %v", err)
// 	}

// 	// Check if the number of bytes written is equal to the length of the JSON representation.
// 	// If it is not, an error is returned.
// 	if c != len(j) {
// 		return errors.New("incorrect number of bytes written")
// 	}

// 	// Return nil if there are no errors.
// 	return nil
// }

// // SaveToFile saves the network's JSON representation to a file.
// // It takes the file path as a parameter and returns an error if there is an error during the saving process.
// func (n *Network) SaveToFile(fp string) error {
// 	// Convert the network to its JSON representation.
// 	j, err := n.ToJson()
// 	if err != nil {
// 		// Return an error with a formatted message if there is an error during the conversion.
// 		return fmt.Errorf("error saving data: %v", err)
// 	}
// 	// Write the JSON representation to the file.
// 	// The os.WriteFile function is used to write the JSON representation to the file.
// 	// It takes the file path, the JSON representation, and the file permission mode as parameters.
// 	// It returns an error if there is an error during the writing process.
// 	return os.WriteFile(fp, j, os.ModePerm)
// }

// // SetDebug sets the debug mode of the network.
// //
// // The debug mode determines whether debug information is printed during the training process.
// //
// // Parameters:
// // - v: A boolean value indicating whether the debug mode is enabled (true) or disabled (false).
// func (n *Network) SetDebug(v bool) {
// 	// Set the debug mode of the network to the specified value.
// 	n.debug = v
// }

// // Debug returns the debug mode of the network.
// //
// // The debug mode determines whether debug information is printed during the training process.
// //
// // Returns:
// // - A boolean value indicating whether the debug mode is enabled (true) or disabled (false).
// func (n *Network) Debug() bool {
// 	// Return the debug mode of the network.
// 	return n.debug
// }

// // FromJson creates a Network object from its JSON representation.
// //
// // This function takes a byte slice containing the JSON representation of a Network object
// // and returns a pointer to the created Network object and an error if there is an error during the creation.
// //
// // Parameters:
// // - b: A byte slice containing the JSON representation of a Network object.
// //
// // Returns:
// // - A pointer to the created Network object.
// // - An error if there is an error during the creation.
// func FromJson(b []byte) (*Network, error) {
// 	// Create a SaveData object to hold the JSON representation.
// 	sd := SaveData{}

// 	// Unmarshal the JSON representation into the SaveData object.
// 	err := json.Unmarshal(b, &sd)
// 	if err != nil {
// 		// Return an error with a formatted message if there is an error during the unmarshalling.
// 		return nil, fmt.Errorf("network unmarshal error: %v", err)
// 	}

// 	// Create a Network object from the SaveData object and return it.
// 	return FromSaveData(&sd)
// }

// // FromSaveData creates a Network object from its SaveData representation.
// //
// // This function takes a pointer to a SaveData object and returns a pointer to the created Network object
// // and an error if there is an error during the creation.
// //
// // Parameters:
// // - sd: A pointer to a SaveData object containing the representation of a Network object.
// //
// // Returns:
// // - A pointer to the created Network object.
// // - An error if there is an error during the creation.
// func FromSaveData(sd *SaveData) (*Network, error) {
// 	// Check if the SaveData object is nil.
// 	if sd == nil {
// 		// Return an error indicating that the SaveData object is missing.
// 		return nil, errors.New("missing save data")
// 	}

// 	// Create slices to hold the weight and bias matrices.
// 	weightMatrices := make([]*Matrix, len(sd.WeightMatrices))
// 	biasMatrices := make([]*Matrix, len(sd.BiasMatrices))

// 	// Iterate through the weight matrices in the SaveData object.
// 	for i, wsd := range sd.WeightMatrices {
// 		// Create a Matrix object from the weight matrix data in the SaveData object.
// 		wm, err := MatrixFromSaveData(wsd)
// 		if err != nil {
// 			// Return an error with a formatted message indicating the error in applying the weight matrix.
// 			return nil, fmt.Errorf("unable to apply weight matrix: %v", err)
// 		}
// 		// Add the created Matrix object to the weightMatrices slice.
// 		weightMatrices[i] = wm
// 	}

// 	// Iterate through the bias matrices in the SaveData object.
// 	for i, bsd := range sd.BiasMatrices {
// 		// Create a Matrix object from the bias matrix data in the SaveData object.
// 		bm, err := MatrixFromSaveData(bsd)
// 		if err != nil {
// 			// Return an error with a formatted message indicating the error in applying the bias matrix.
// 			return nil, fmt.Errorf("unable to apply bias matrix: %v", err)
// 		}
// 		// Add the created Matrix object to the biasMatrices slice.
// 		biasMatrices[i] = bm
// 	}

// 	// Create a slice to hold the value matrices.
// 	valueMatrices := make([]*Matrix, len(sd.Topology))

// 	// Iterate through the topology in the SaveData object.
// 	for i, t := range sd.Topology {
// 		// Create a new Matrix object with the specified size and add it to the valueMatrices slice.
// 		valueMatrices[i] = NewMatrix(t, 1)
// 	}

// 	// Get the function name from the SaveData object.
// 	fn := FunctionName(sd.Functions)

// 	// Get the corresponding Functions struct from the FunctionList based on the function name.
// 	f := FunctionList[fn]

// 	// Create a new Network object with the specified values and return it.
// 	n := Network{
// 		topology:       sd.Topology,
// 		learningRate:   sd.LearningRate,
// 		functionName:   fn,
// 		activation:     f.Activation,
// 		derivative:     f.Derivative,
// 		weightMatrices: weightMatrices,
// 		valueMatrices:  valueMatrices,
// 		biasMatrices:   biasMatrices,
// 	}

// 	return &n, nil
// }

// // Read reads the network's JSON representation from the provided reader.
// // It takes an io.Reader as a parameter and returns a pointer to the created Network object
// // and an error if there is an error during the creation.
// func Read(r io.Reader) (*Network, error) {
// 	// Create a buffer to hold the data read from the reader.
// 	buf := make([]byte, 0, 64)
// 	// Create a slice to hold the final data.
// 	result := make([]byte, 0)
// 	// Create a variable to keep track of the total number of bytes read.
// 	total := 0
// 	// Loop until there is no more data to read.
// 	for {
// 		// Read data from the reader.
// 		count, err := r.Read(buf)
// 		// If there is an error during the reading process, return an error.
// 		if err != nil {
// 			return nil, fmt.Errorf("read error: %v", err)
// 		}
// 		// If the number of bytes read is greater than 0, update the total count.
// 		if count > 0 {
// 			total += count
// 			// Append the read data to the result slice.
// 			result = append(result, buf[:count]...)
// 		}
// 		// If the number of bytes read is less than the size of the buffer,
// 		// it means there is no more data to read, so break the loop.
// 		if count < len(buf) {
// 			break
// 		}
// 	}
// 	// Create a Network object from the JSON representation and return it.
// 	return FromJson(result)
// }

// // FromFile reads the network's JSON representation from a file and returns a
// // pointer to the created Network object and an error if there is an error during
// // the creation. The function takes a file path as a parameter and returns a pointer
// // to the created Network object and an error if there is an error during the creation.
// func FromFile(fp string) (*Network, error) {
// 	// Read the file and return an error if there is an error during the reading process.
// 	b, err := os.ReadFile(fp)
// 	if err != nil {
// 		return nil, fmt.Errorf("unable to read data: %v", err)
// 	}
// 	// Create a Network object from the JSON representation and return it.
// 	return FromJson(b)
// }
