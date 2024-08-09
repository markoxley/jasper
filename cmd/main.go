package main

import (
	"log"
	"time"

	v1 "github.com/markoxley/jasper/v1"
	v2 "github.com/markoxley/jasper/v2"
)

const (
	epochs       = 1000000
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
	for idx := range i {
		td.AddRow(i[idx], o[idx])
	}
	nc := v1.NewConfig(t)
	nn, err := v1.New(nc)
	if err != nil {
		panic(err)
	}
	errValue, err := nn.Train(td)
	if err != nil {
		panic(err)
	}
	log.Printf("Version1 error value: %v\n", errValue)
	return time.Since(start)
}

func version2(i, o [][]float64, t []uint32) time.Duration {
	start := time.Now()
	td := v2.NewTrainingData(epochs, split, learningrate)
	for idx := range i {
		td.AddRow(i[idx], o[idx])
	}
	nc := v2.NewConfig(t)
	nn, err := v2.New(nc)
	if err != nil {
		panic(err)
	}
	errValue, err := nn.Train(td)
	if err != nil {
		panic(err)
	}
	log.Printf("Version2 error value: %v\n", errValue)
	return time.Since(start)
}
func main() {
	data := [][]float64{}
	for i := 0; i < 10000; i++ {
		data = append(data, org...)
	}
	log.Printf("%d data records\n", len(data))
	inputs, outputs := splitData(data)
	topology := []uint32{3, 4, 1}
	log.Println("Starting version 1...")
	d1 := version1(inputs, outputs, topology)
	log.Printf("Version1: %v seconds\n", d1.Seconds())
	log.Println("Starting version 2...")
	d2 := version2(inputs, outputs, topology)
	log.Printf("Version2: %v seconds\n", d2.Seconds())

}
