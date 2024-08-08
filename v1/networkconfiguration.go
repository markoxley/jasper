package jasper

type NetworkConfiguration struct {
	Topology     []uint32
	LearningRate float64
	Functions    FunctionName
	Quiet        bool
}

func NewConfig(topology []uint32) *NetworkConfiguration {
	return &NetworkConfiguration{
		Topology:     topology,
		LearningRate: 0.1,
		Functions:    Sigmoid,
		Quiet:        true,
	}
}
