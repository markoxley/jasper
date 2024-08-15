package jasper

// NetworkConfiguration represents the configuration of a neural network.
// It contains the topology of the network, the learning rate, activation and output functions,
// quiet mode, softmax mode, and the error function.
type NetworkConfiguration struct {
	// Topology is a slice of uint32 representing the topology of the neural network.
	// The topology is a sequence of integers where each integer represents the number of neurons in a layer.
	Topology []uint32

	// LearningRate is a float64 representing the learning rate of the network.
	// The learning rate determines how quickly the weights of the network are adjusted during training.
	LearningRate float64

	// Activation is an enum representing the activation function used in the hidden layers of the network.
	Activation ActivationFunction

	// Output is an enum representing the activation function used in the output layer of the network.
	Output ActivationFunction

	// Quiet is a boolean indicating whether the network should run in quiet mode.
	// If true, the network will not print any messages during training.
	Quiet bool

	// SoftMax is a boolean indicating whether the network should use the SoftMax activation function in the output layer.
	// If true, the output is normalized to a probability distribution.
	SoftMax bool

	// Error is an enum representing the error function used in the network.
	// The error function is used to calculate the error between the predicted output and the target output.
	Error ErrorFunction
}

// NewConfig creates a new NetworkConfiguration object with the given topology.
// It sets the default learning rate to 0.1, the default activation function to Sigmoid,
// and the default quiet mode to true.
//
// Parameters:
// - topology: A slice of uint32 representing the topology of the neural network.
//
// Returns:
// - A pointer to the created NetworkConfiguration object.
func NewConfig(topology []uint32) *NetworkConfiguration {
	// Create a new NetworkConfiguration object and set its properties.
	return &NetworkConfiguration{
		Topology:     topology, // Set the topology of the neural network.
		LearningRate: 0.1,      // Set the learning rate to 0.1.
		Activation:   Sigmoid,  // Set the activation function to Sigmoid.
		Output:       Sigmoid,
		Quiet:        false, // Set the quiet mode to true.
		Error:        MeanSquaredError,
		SoftMax:      false,
	}
}
