package main

import (
	"fmt"
	"log"
	"math"
	"time"

	v1 "github.com/markoxley/jasper/v1"
)

const (
	epochs       = 10_000
	split        = 0.8
	learningrate = 0.01
)

var (
	org = [][]float64{
		{0, 0, 0, 0},
		{0, 1, 0, 0},
		{1, 0, 0, 0},
		{1, 1, 0, 1},
		{0, 0, 1, 0},
		{0, 1, 1, 1},
		{1, 0, 1, 1},
		{1, 1, 1, 1},
		{0, 0, 2, 0},
		{0, 1, 2, 1},
		{1, 0, 2, 1},
		{1, 1, 2, 0},
		{0, 0, 3, 1},
		{0, 1, 3, 1},
		{1, 0, 3, 1},
		{1, 1, 3, 0},
		{0, 0, 4, 1},
		{0, 1, 4, 0},
		{1, 0, 4, 0},
		{1, 1, 4, 0},
	}

	calcs = []string{"AND", "OR", "XOR", "NAND", "NOR"}
)

func splitData(data [][]float64) ([][]float64, [][]float64) {
	inputs := make([][]float64, len(data))
	outputs := make([][]float64, len(data))
	for i, d := range data {
		inputs[i] = d[:3]
		outputs[i] = d[3:]
	}
	return inputs, outputs
}
func version1(i, o [][]float64, t []uint32) time.Duration {
	start := time.Now()
	td := v1.NewTrainingData(epochs, split, learningrate)
	td.TargetError = 0.2
	for idx := range i {
		td.AddRow(i[idx], o[idx])
	}
	nc := v1.NewConfig(t)
	nc.Error = v1.MeanSquaredError
	nc.Activation = v1.Sigmoid
	nc.Output = v1.Sigmoid
	nn, err := v1.New(nc)
	if err != nil {
		panic(err)
	}
	errValue, err := nn.Train(td)
	if err != nil {
		panic(err)
	}
	log.Printf("Version1 error value: %v\n", errValue)
	success := true
	calcType := ""
	for _, o := range org {
		nct := calcs[int(o[2])]
		if nct != calcType {
			calcType = nct
			fmt.Printf("Comparison: %v\n", calcType)
		}
		v, _ := nn.Predict(o[:3])
		ok := "FALSE"
		result := int(math.Round(v[0]))
		if result == int(o[3]) {
			ok = "TRUE"
		}
		if ok == "FALSE" {
			success = false
		}
		fmt.Printf("\t%v = %v\tActual = %v\t%s\n", o[:3], o[3], result, ok)
	}
	if success {
		fmt.Println("Training successful")
	} else {
		fmt.Println("Training failed")
	}
	return time.Since(start)
}
func main() {
	data := [][]float64{}
	for i := 0; i < 10; i++ {
		data = append(data, org...)
	}
	log.Printf("%d data records\n", len(data))
	inputs, outputs := splitData(data)
	topology := []uint32{3, 6, 1}
	log.Println("Starting version 1...")
	d1 := version1(inputs, outputs, topology)
	log.Printf("Version1: %v seconds\n", d1.Seconds())

}
