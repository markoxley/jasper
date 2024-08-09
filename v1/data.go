package jasper

import (
	"math"
	"math/rand"
)

type DataRow struct {
	Input []float64
	Ouput []float64
}

type TrainingData struct {
	trainingData []*DataRow
	testingData  []*DataRow
	Data         []*DataRow
	Split        float64
	Iterations   uint32
	TargetError  float64
	position     int
}

// NewTrainingData creates a new instance of the TrainingData type.
//
// iterations specifies the number of iterations for the training data.
// split specifies the proportion of the data to be used for training.
// errMargin specifies the target error margin for the training data.
// Returns a pointer to the newly created TrainingData instance.
func NewTrainingData(iterations uint32, split float64, errMargin float64) *TrainingData {
	return &TrainingData{
		Split:       split,
		Iterations:  iterations,
		TargetError: errMargin,
	}
}

// AddRow adds a new data row to the training data set.
//
// inputs is the input data for the row, and output is the corresponding output data.
// No return value.
func (d *TrainingData) AddRow(inputs, output []float64) {
	d.Data = append(d.Data, &DataRow{
		Input: inputs,
		Ouput: output,
	})
}

// Prepare prepares the training data by splitting it into training and testing
// data sets based on the specified split value.
//
// The training data set will contain approximately split * len(d.Data) rows.
// The testing data set will contain the remaining rows.
//
// No return value.
func (d *TrainingData) Prepare() {
	// Determine the number of rows to be used for training
	trainCount := int(math.Round(float64(len(d.Data)) * d.Split))
	// Determine the number of rows to be used for testing
	testCount := len(d.Data) - trainCount

	// Create slices with the appropriate capacities to hold the training and testing data
	d.trainingData = make([]*DataRow, 0, trainCount)
	d.testingData = make([]*DataRow, 0, testCount)

	// Create a slice to hold the indices of the data rows
	index := make([]int, len(d.Data))
	for i := range index {
		index[i] = i
	}

	// Shuffle the indices to randomize the order of the data rows
	for i := 0; i < len(d.Data); i++ {
		p1 := rand.Intn(len(d.Data))
		p2 := rand.Intn(len(d.Data))
		tmp := index[p1]
		index[p1] = index[p2]
		index[p2] = tmp
	}

	// Append the data rows to the appropriate slice based on their index
	for i, idx := range index {
		if i < trainCount {
			// Append the row to the training data slice
			d.trainingData = append(d.trainingData, d.Data[idx])
		} else {
			// Append the row to the testing data slice
			d.testingData = append(d.testingData, d.Data[idx])
		}
	}

	// Reset the position counter
	d.position = 0
}

// RandomTrainingRow returns a random training data row from the training data slice.
//
// It does this by generating a random index between 0 and the length of the training data slice
// and returning the data row at that index.
//
// Returns a pointer to a DataRow struct.
func (d *TrainingData) RandomTrainingRow() *DataRow {
	// Generate a random index between 0 and the length of the training data slice
	randomIndex := rand.Intn(len(d.trainingData))

	// Return the data row at the random index
	return d.trainingData[randomIndex]
}

// NextRow returns the next training data row from the training data slice.
//
// If the current position is greater than or equal to the length of the training
// data slice, it resets the position to 0 and returns nil. Otherwise, it returns
// the data row at the current position and increments the position for the next
// call to NextRow.
//
// Returns a pointer to a DataRow struct.
func (d *TrainingData) NextRow() *DataRow {
	// If the current position is beyond the length of the training data slice,
	// reset the position to 0 and return nil.
	if d.position >= len(d.trainingData) {
		d.position = 0
		return nil
	}

	// Defer incrementing the position for the next call to NextRow.
	defer func() {
		d.position++
	}()

	// Return the data row at the current position.
	return d.trainingData[d.position]
}

// TestData returns the testing data slice.
//
// It contains the data rows that are not used for training.
//
// Returns a slice of pointers to DataRow structs.
func (d *TrainingData) TestData() []*DataRow {
	// Return the testing data slice.
	return d.testingData
}

// TrainingCount returns the number of training rows in the TrainingData struct.
//
// This function returns the length of the trainingData slice, which contains
// the rows of data used for training.
//
// Returns an integer representing the number of training rows.
func (d *TrainingData) TrainingCount() int {
	// Return the length of the trainingData slice, which contains the rows of
	// data used for training.
	return len(d.trainingData)
}

// TestCount returns the number of testing rows in the TrainingData struct.
//
// This function returns the length of the testingData slice, which contains
// the rows of data not used for training.
//
// Returns an integer representing the number of testing rows.
func (d *TrainingData) TestCount() int {
	// Return the length of the testingData slice, which contains the rows of
	// data not used for training.
	return len(d.testingData)
}
