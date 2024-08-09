package jasper

type NetworkConfiguration struct {
	Topology     []uint32
	LearningRate float64
	Functions    FunctionName
	Quiet        bool
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
		Functions:    Sigmoid,  // Set the activation function to Sigmoid.
		Quiet:        true,     // Set the quiet mode to true.
	}
}
