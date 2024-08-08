package jasper

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"time"
)

type SaveData struct {
	Topology       []uint32          `json:"t"`
	WeightMatrices []*MatrixSaveData `json:"w"`
	BiasMatrices   []*MatrixSaveData `json:"b"`
	LearningRate   float64           `json:"l"`
	Functions      uint32            `json:"f"`
}
type Network struct {
	topology       []uint32
	weightMatrices []*Matrix
	valueMatrices  []*Matrix
	biasMatrices   []*Matrix
	learningRate   float64
	functionName   FunctionName
	activation     NeuralFunction
	derivative     NeuralFunction
	debug          bool
}

func getRandomFloats(sz int) []float64 {
	r := make([]float64, sz)
	for i, _ := range r {
		r[i] = ApplyRandom(0)
	}
	return r
}

// NewSimple creates a new simple neural network
func New(c *NetworkConfiguration) (*Network, error) {
	s := Network{
		topology:     c.Topology,
		learningRate: c.LearningRate,
		functionName: c.Functions,
		activation:   FunctionList[c.Functions].Activation,
		derivative:   FunctionList[c.Functions].Derivative,
		debug:        !c.Quiet,
	}
	for i := 0; i < len(s.topology)-1; i++ {
		wm := NewMatrix(s.topology[i+1], s.topology[i])
		s.weightMatrices = append(s.weightMatrices, wm.ApplyFunction(ApplyRandom))

		bm := NewMatrix(s.topology[i+1], 1)
		s.biasMatrices = append(s.biasMatrices, bm.ApplyFunction(ApplyRandom))
	}
	s.valueMatrices = make([]*Matrix, len(s.topology))
	return &s, nil
}

// FeedForward runs the network forwards
func (n *Network) feedForward(input []float64) error {
	if len(input) != int(n.topology[0]) {
		return errors.New("incorrect input size")
	}
	values := NewMatrix(uint32(len(input)), 1)
	for i, in := range input {
		values.Set(uint32(i), 0, in)
	}
	var err error
	// feed forward to next layer
	for i, w := range n.weightMatrices {
		n.valueMatrices[i] = values
		values, err = values.Multiply(w)
		if err != nil {
			return fmt.Errorf("feed forward error: %v", err)
		}
		values, err = values.Add(n.biasMatrices[i])
		if err != nil {
			return fmt.Errorf("feed forward error: %v", err)
		}
		values = values.ApplyFunction(n.activation)
	}
	n.valueMatrices[len(n.weightMatrices)] = values

	return nil
}

func (n *Network) backPropagate(tgtOut []float64) error {
	if len(tgtOut) != int(n.topology[len(n.topology)-1]) {
		return errors.New("output is incorrect size")
	}
	errMtx := NewMatrix(uint32(len(tgtOut)), 1)
	errMtx.SetValues(tgtOut)
	errMtx, err := errMtx.Add(n.valueMatrices[len(n.valueMatrices)-1].Negative())
	if err != nil {
		return fmt.Errorf("back propagation error: %v", err)
	}

	for i := len(n.weightMatrices) - 1; i >= 0; i-- {
		prevErrors, err := errMtx.Multiply(n.weightMatrices[i].Transpose())
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}
		dOutputs := n.valueMatrices[i+1].ApplyFunction(n.derivative)
		gradients, err := errMtx.MultiplyElements(dOutputs)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}
		gradients = gradients.MultiplyScalar(n.learningRate)
		weightGradients, err := n.valueMatrices[i].Transpose().Multiply(gradients)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}
		n.weightMatrices[i], err = n.weightMatrices[i].Add(weightGradients)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}
		n.biasMatrices[i], err = n.biasMatrices[i].Add(gradients)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}
		errMtx = prevErrors
	}
	return nil

}

func (n *Network) getPrediction() []float64 {
	return n.valueMatrices[len(n.valueMatrices)-1].Values()
}

func (n *Network) Train(td *TrainingData) (float64, error) {
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
	iterCount := 0
	for i := 0; i < int(td.Iterations); i++ {
		if n.debug {
			iterCount++
			if i%1000 == 0 {
				if i > 0 && i%80_000 == 0 {
					fmt.Printf(" %v\n", i)
				}
				fmt.Print(".")

			}
		}
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
			if math.Sqrt(v) > td.TargetError {
				errorWithinTolerence = false
				break
			}
			errSum += v
		}
		errSum = math.Sqrt(errSum / float64(len(td.TestData())))
		if errorWithinTolerence && errSum <= td.TargetError {
			if n.debug {
				fmt.Print("\nterminating early. Within tolerance.")
			}
			break
		}
	}
	if n.debug {
		stop := time.Now()
		fmt.Printf("\ntraining complete at %v\n", stop)
		fmt.Printf("training took %v minutes\n", stop.Sub(start).Minutes())
		fmt.Printf("\t%v iterations run\n", iterCount)
		fmt.Printf("\terror margin is %v\n", errSum)
	}
	return errSum, nil
}

func (n *Network) Predict(input []float64) ([]float64, error) {
	err := n.feedForward(input)
	if err != nil {
		return nil, fmt.Errorf("prediction error: %v", err)
	}
	return n.getPrediction(), nil
}

func (n *Network) ToSaveData() *SaveData {
	sd := SaveData{
		Topology:       n.topology,
		LearningRate:   n.learningRate,
		WeightMatrices: make([]*MatrixSaveData, len(n.weightMatrices)),
		BiasMatrices:   make([]*MatrixSaveData, len(n.biasMatrices)),
		Functions:      uint32(n.functionName),
	}
	for i, wm := range n.weightMatrices {
		sd.WeightMatrices[i] = wm.ToSaveData()
	}
	for i, bm := range n.biasMatrices {
		sd.BiasMatrices[i] = bm.ToSaveData()
	}
	return &sd
}

func (n *Network) ToJson() ([]byte, error) {
	return json.Marshal(n.ToSaveData())
}

func (n *Network) Write(w io.Writer) error {
	j, err := n.ToJson()
	if err != nil {
		return fmt.Errorf("network write error: %v", err)
	}
	c, err := w.Write(j)
	if err != nil {
		return fmt.Errorf("network write error: %v", err)
	}
	if c != len(j) {
		return errors.New("incorrect number of bytes written")
	}
	return nil
}

func (n *Network) SaveToFile(fp string) error {
	j, err := n.ToJson()
	if err != nil {
		return fmt.Errorf("error saving data: %v", err)
	}
	return os.WriteFile(fp, j, os.ModePerm)
}

func (n *Network) SetDebug(v bool) {
	n.debug = v
}

func (n *Network) Debug() bool {
	return n.debug
}

func FromJson(b []byte) (*Network, error) {
	sd := SaveData{}
	err := json.Unmarshal(b, &sd)
	if err != nil {
		return nil, fmt.Errorf("network unmarshal error: %v", err)
	}

	return FromSaveData(&sd)
}

func FromSaveData(sd *SaveData) (*Network, error) {
	if sd == nil {
		return nil, errors.New("missing save data")
	}
	weightMatrices := make([]*Matrix, len(sd.WeightMatrices))
	for i, wsd := range sd.WeightMatrices {
		wm, err := MatrixFromSaveData(wsd)
		if err != nil {
			return nil, fmt.Errorf("unable to apply weight matrix: %v", err)
		}
		weightMatrices[i] = wm
	}

	biasMatrices := make([]*Matrix, len(sd.BiasMatrices))
	for i, bsd := range sd.BiasMatrices {
		bm, err := MatrixFromSaveData(bsd)
		if err != nil {
			return nil, fmt.Errorf("unable to apply bias matrix: %v", err)
		}
		biasMatrices[i] = bm
	}

	valueMatrices := make([]*Matrix, len(sd.Topology))
	for i, t := range sd.Topology {
		valueMatrices[i] = NewMatrix(t, 1)
	}

	fn := FunctionName(sd.Functions)
	f := FunctionList[fn]
	n := Network{
		topology:       sd.Topology,
		learningRate:   sd.LearningRate,
		functionName:   fn,
		activation:     f.Activation,
		derivative:     f.Derivative,
		weightMatrices: weightMatrices,
		valueMatrices:  valueMatrices,
		biasMatrices:   biasMatrices,
	}

	return &n, nil
}

func Read(r io.Reader) (*Network, error) {
	b := make([]byte, 0, 64)
	res := make([]byte, 0)
	t := 0
	for {
		c, err := r.Read(b)
		if err != nil {
			return nil, fmt.Errorf("read error: %v", err)
		}
		if c > 0 {
			t += c
			res = append(res, b[:c]...)
		}
		if c < len(b) {
			break
		}

	}
	return FromJson(res)
}

func FromFile(fp string) (*Network, error) {
	b, err := os.ReadFile(fp)
	if err != nil {
		return nil, fmt.Errorf("unable to read data: %v", err)
	}
	return FromJson(b)
}
