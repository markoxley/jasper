# Jasper

## Basic artificial neural network

This is a basic artificial neural network, writtin entirely in Go.

This implementation allows for multiple hidden layers.

No other dependencies have been used.

_Disclaimer: this was created as an experiment and purely for educational purposes. I make no claim that this library is production ready._

### Why Jasper?

Why not?

### Installation

```sh
    go get github.com/markoxley/jasper
```

### Usage

Import the current version into your source:

```go
    import github.com/markoxley/jasper/v1
```

```go
    // Create configuration for the network
    // In this case, we have 13 input neurons, 11
    // output neurons and one hidden layer with 16 neurons
    config := jasper.NewConfiguration[]uint32{13, 16, 11})

    // Create a new instance of the neural network,
    // passing the configurations
    nn, _ := jasper.New(config)

    // Initialis a training data set
    // This data set will attempt 100,000 iterations,
    // using 50% of the test data, and a target error
    // tolerance of 0.1
    td := jasper.NewTrainingData(100_000, 0.5, 0.1)


    // Add the training and test data to the data set
    for i := 0; i < len(targetInputs); i++ {
    	td.AddRow(targetInputs[i], targetOutputs[i])
    }

    // Train the network, handling any errors
    if _, err := nn.Train(td); err != nil {
		return fmt.Errorf("training error: %v", err)
	}

    // Get a prediction from the network, again, be
    // mindful of any returned errors
    pr, err := nn.Predict(in)
```
