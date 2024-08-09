package jasper

import (
	"errors"
	"fmt"
)

type MatrixSaveData struct {
	Cols   uint32    `json:"c"`
	Rows   uint32    `json:"r"`
	Values []float64 `json:"v"`
}

// Matrix holds the matrix data type
type Matrix struct {
	cols   uint32
	rows   uint32
	values []float64
}

// NewMatrix creates a new Matrix with the specified number of columns and rows.
//
// cols: The number of columns in the matrix.
// rows: The number of rows in the matrix.
//
// Returns a pointer to the newly created Matrix.
func NewMatrix(cols, rows uint32) *Matrix {
	// Create a new Matrix struct with the specified columns and rows.
	// Initialize the values slice with the product of cols and rows.
	m := Matrix{
		cols:   cols,
		rows:   rows,
		values: make([]float64, cols*rows),
	}

	// Return a pointer to the newly created Matrix.
	return &m
}

// NewFromSlice creates a new Matrix from a slice of float64 values.
// The Matrix will have one column and the same number of rows as the length of the slice.
//
// slc: A slice of float64 values to create the Matrix from.
//
// Returns a pointer to the newly created Matrix.
func NewFromSlice(slc []float64) *Matrix {
	// Create a new Matrix with one column and the same number of rows as the length of the slice.
	m := NewMatrix(uint32(len(slc)), 1)

	// Set the values of the Matrix to the values in the slice.
	m.values = slc

	// Return a pointer to the newly created Matrix.
	return m
}

// ApplyFunction applies a function to each element of the matrix
//
// Parameters:
// - f: A function that takes a float64 value and returns a float64 value.
//
// Returns:
//   - A new matrix with the same dimensions as the original matrix, but with
//     the function applied to each element.
//
// ApplyFunction appies a function to the matrix elements
func (m *Matrix) ApplyFunction(f NeuralFunction) *Matrix {
	// Create a new matrix with the same dimensions as the original matrix.
	o := NewMatrix(m.cols, m.rows)

	// Iterate over each element of the original matrix.
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			// Get the value of the current element.
			mC, _ := m.At(x, y)

			// Apply the function to the value and set the value in the new matrix.
			o.Set(x, y, f(mC))
		}
	}

	// Return the new matrix.
	return o
}

// Cols returns the number of columns in the matrix. This is a getter method
// that returns the value of the private field 'cols'.
//
// Returns:
// - The number of columns in the matrix (uint32).
// Cols returns the number of columns in the matrix
func (m *Matrix) Cols() uint32 {
	return m.cols // Return the number of columns in the matrix
}

// Rows returns the number of rows in the matrix. This is a getter method that
// returns the value of the private field 'rows'.
//
// Returns:
// - The number of rows in the matrix (uint32).
func (m *Matrix) Rows() uint32 {
	return m.rows // Return the number of rows in the matrix
}

// Values returns a slice of all the values in the matrix.
//
// This function returns a reference to the private field 'values' of the Matrix
// struct. This means that any changes made to the returned slice will be
// reflected in the Matrix.
//
// Returns:
// - A slice of float64 values representing the values in the matrix.
func (m *Matrix) Values() []float64 {
	// Return a reference to the private field 'values' of the Matrix struct.
	return m.values
}

// At returns the value at the specified column and row of the matrix.
//
// Parameters:
// - col: The column index of the value to retrieve (uint32).
// - row: The row index of the value to retrieve (uint32).
//
// Returns:
// - The value at the specified column and row of the matrix (float64).
// - An error if the column or row index is out of range (error).
func (m *Matrix) At(col, row uint32) (float64, error) {
	// Check if the column index is out of range.
	if col >= m.cols {
		return 0, fmt.Errorf("column out of range: %v maximum, %v requested", m.cols-1, col)
	}

	// Check if the row index is out of range.
	if row >= m.rows {
		return 0, fmt.Errorf("row out of range: %v maximum, %v requested", m.rows-1, row)
	}

	// Return the value at the specified column and row of the matrix.
	return m.values[row*m.cols+col], nil
}

// Set assigns a value to the specified cell at column, row
//
// Parameters:
// - col: The column index of the cell to set (uint32).
// - row: The row index of the cell to set (uint32).
// - v: The value to assign to the cell (float64).
//
// Returns:
// - An error if the column or row index is out of range (error).
func (m *Matrix) Set(col, row uint32, v float64) error {
	// Check if the column index is out of range.
	if col >= m.cols {
		return fmt.Errorf("column out of range: %v maximum, %v requested", m.cols-1, col)
	}

	// Check if the row index is out of range.
	if row >= m.rows {
		return fmt.Errorf("row out of range: %v maximum, %v requested", m.rows-1, row)
	}

	// Assign the value to the specified cell.
	m.values[row*m.cols+col] = v

	return nil
}

// Multiply multiplies the matrix with another matrix, returning a new matrix
//
// The function performs matrix multiplication between the receiver matrix and the
// target matrix. It checks if the dimensions of the matrices are compatible for
// multiplication and returns an error if they are not. If the dimensions are
// compatible, the function creates a new matrix to store the result of the
// multiplication and performs the multiplication element by element.
//
// Parameters:
// - tgt: The target matrix to multiply with the receiver matrix (Matrix).
//
// Returns:
// - The resulting matrix after matrix multiplication (Matrix).
// - An error if the dimensions of the matrices are not compatible (error).
func (m *Matrix) Multiply(tgt *Matrix) (*Matrix, error) {
	// Check if the receiver matrix's number of columns is equal to the target
	// matrix's number of rows. If not, return an error.
	if m.cols != tgt.rows {
		return nil, errors.New("shape error")
	}

	// Create a new matrix to store the result of the multiplication. The number of
	// columns of the new matrix is equal to the number of columns of the target
	// matrix, and the number of rows is equal to the number of rows of the receiver
	// matrix.
	o := NewMatrix(tgt.cols, m.rows)

	// Perform matrix multiplication element by element.
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			var v float64
			// Iterate over the number of columns of the receiver matrix.
			for k := uint32(0); k < m.cols; k++ {
				// Get the value at the current column and row of the receiver matrix.
				mC, _ := m.At(k, y)
				// Get the value at the current column and row of the target matrix.
				tC, _ := tgt.At(x, k)
				// Update the value for the current element of the resulting matrix.
				v += mC * tC
			}
			// Set the value of the resulting matrix at the current column and row.
			o.Set(x, y, v)
		}
	}

	// Return the resulting matrix and no error.
	return o, nil
}

// MultiplyScalar multiplies each element of the matrix by a scalar value.
//
// Parameters:
// - v: The scalar value to multiply each element of the matrix by.
//
// Returns:
// - A new matrix with each element multiplied by the scalar value.
func (m *Matrix) MultiplyScalar(v float64) *Matrix {
	// Create a new matrix with the same dimensions as the receiver matrix.
	o := NewMatrix(m.cols, m.rows)

	// Iterate over each element of the receiver matrix and multiply it by the
	// scalar value.
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			// Get the value at the current column and row of the receiver matrix.
			mC, _ := m.At(x, y)

			// Set the value of the resulting matrix at the current column and row to
			// the product of the value from the receiver matrix and the scalar value.
			o.Set(x, y, v*mC)
		}
	}

	// Return the resulting matrix.
	return o
}

// MultiplyElements multiplies each element of the matrix with the corresponding
// element in the target matrix, returning a new matrix.
//
// Parameters:
// - tgt: The target matrix to multiply with.
//
// Returns:
//   - A new matrix with each element multiplied by the corresponding element in
//     the target matrix.
//   - An error if the shapes of the matrices are not the same.
func (m *Matrix) MultiplyElements(tgt *Matrix) (*Matrix, error) {
	// Check if the shapes of the matrices are the same.
	if m.cols != tgt.cols || m.rows != tgt.rows {
		return nil, errors.New("shape error")
	}

	// Create a new matrix with the same dimensions as the receiver matrix.
	o := NewMatrix(m.cols, m.rows)

	// Iterate over each element of the receiver matrix and multiply it by the
	// corresponding element in the target matrix.
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			// Get the value at the current column and row of the receiver matrix.
			mC, _ := m.At(x, y)

			// Get the value at the current column and row of the target matrix.
			tC, _ := tgt.At(x, y)

			// Set the value of the resulting matrix at the current column and row to
			// the product of the values from the receiver and target matrices.
			o.Set(x, y, mC*tC)
		}
	}

	// Return the resulting matrix and no error.
	return o, nil
}

// Add adds two matrices element-wise, returning a new matrix.
//
// Parameters:
// - tgt: The target matrix to add with.
//
// Returns:
//   - A new matrix with each element being the sum of the corresponding elements
//     from the receiver and target matrices.
//   - An error if the shapes of the matrices are not the same.
func (m *Matrix) Add(tgt *Matrix) (*Matrix, error) {
	// Check if the shapes of the matrices are the same.
	if m.cols != tgt.cols || m.rows != tgt.rows {
		return nil, errors.New("shape error")
	}

	// Create a new matrix with the same dimensions as the receiver matrix.
	o := NewMatrix(m.cols, m.rows)

	// Iterate over each element of the receiver matrix and add the corresponding
	// element from the target matrix.
	for y := uint32(0); y < m.rows; y++ {
		for x := uint32(0); x < m.cols; x++ {
			// Get the value at the current column and row of the receiver matrix.
			mC, _ := m.At(x, y)

			// Get the value at the current column and row of the target matrix.
			tC, _ := tgt.At(x, y)

			// Set the value of the resulting matrix at the current column and row to
			// the sum of the values from the receiver and target matrices.
			o.Set(x, y, mC+tC)
		}
	}

	// Return the resulting matrix and no error.
	return o, nil
}

// AddScalar adds a scalar value to each element of the matrix.
//
// Parameters:
// - v: The scalar value to add to each element of the matrix.
//
// Returns:
//   - A new matrix with each element being the sum of the corresponding element
//     from the receiver matrix and the scalar value.
func (m *Matrix) AddScalar(v float64) *Matrix {
	// Create a new matrix with the same dimensions as the receiver matrix.
	o := NewMatrix(m.cols, m.rows)

	// Iterate over each element of the receiver matrix and add the scalar value.
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			// Get the value at the current column and row of the receiver matrix.
			mC, _ := m.At(x, y)

			// Set the value of the resulting matrix at the current column and row to
			// the sum of the value from the receiver matrix and the scalar value.
			o.Set(x, y, v+mC)
		}
	}

	// Return the resulting matrix.
	return o
}

// Negative returns a new matrix with the opposite values of the receiver matrix
//
// Parameters:
//
//	None
//
// Returns:
//   - A new matrix with the opposite values of the receiver matrix
func (m *Matrix) Negative() *Matrix {
	// Create a new matrix with the same dimensions as the receiver matrix.
	o := NewMatrix(m.cols, m.rows)

	// Iterate over each element of the receiver matrix and set the value of the
	// corresponding element in the new matrix to its opposite.
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			// Get the value at the current column and row of the receiver matrix.
			mC, _ := m.At(x, y)

			// Set the value of the resulting matrix at the current column and row to
			// the opposite of the value from the receiver matrix.
			o.Set(x, y, -mC)
		}
	}

	// Return the new matrix.
	return o
}

// Transpose returns a new matrix that is the transpose of the receiver matrix
//
// Parameters:
//
//	None
//
// Returns:
//   - A new matrix that is the transpose of the receiver matrix
func (m *Matrix) Transpose() *Matrix {
	// Create a new matrix with dimensions reversed from the receiver matrix.
	o := NewMatrix(m.rows, m.cols)

	// Iterate over each element of the receiver matrix.
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			// Get the value at the current row and column of the receiver matrix.
			mC, _ := m.At(y, x)

			// Set the value of the corresponding element in the new matrix, which is
			// at the current column and row of the receiver matrix.
			o.Set(x, y, mC)
		}
	}

	// Return the new matrix.
	return o
}

// SetValues sets the values of the matrix to the provided slice of float64 values
//
// Parameters:
//   - vals: A slice of float64 values that will be used to set the values of the matrix
//
// Returns:
//   - An error if the provided slice is nil or the wrong size
func (m *Matrix) SetValues(vals []float64) error {
	// Check if the provided slice is nil
	if vals == nil {
		// Return an error message indicating that the values are missing
		return errors.New("missing values")
	}
	// Check if the provided slice is the wrong size
	if len(vals) != len(m.values) {
		// Return an error message indicating that the size is incorrect
		return errors.New("size error")
	}
	// Set the values of the matrix to the provided slice
	m.values = vals
	// Return nil to indicate success
	return nil
}

// ToSaveData converts the matrix to a MatrixSaveData object, which can be used for serialization.
//
// This function returns a pointer to a MatrixSaveData object, which contains the
// columns, rows, and values of the matrix.
//
// Returns:
//   - A pointer to a MatrixSaveData object.
func (m *Matrix) ToSaveData() *MatrixSaveData {
	// Create a new MatrixSaveData object and set its fields to the corresponding fields of the matrix.
	sd := MatrixSaveData{
		Cols:   m.cols,   // Set the columns field to the value of the matrix's cols field.
		Rows:   m.rows,   // Set the rows field to the value of the matrix's rows field.
		Values: m.values, // Set the values field to the value of the matrix's values field.
	}

	// Return a pointer to the newly created MatrixSaveData object.
	return &sd
}


// MatrixFromSaveData converts a MatrixSaveData object back into a Matrix object.
// This is useful for deserialization.
//
// Parameters:
//   - sd: A pointer to a MatrixSaveData object containing the matrix data.
//
// Returns:
//   - A pointer to a newly created Matrix object with the same data as the input.
//   - An error if the input is nil or the matrix data is invalid.
func MatrixFromSaveData(sd *MatrixSaveData) (*Matrix, error) {
	// Check if the input is nil
	if sd == nil {
		// Return an error indicating that the input is missing
		return nil, errors.New("missing save data")
	}

	// Create a new Matrix object and set its fields to the corresponding fields of the input
	m := Matrix{
		cols:   sd.Cols,   // Set the columns field to the value of the input's Cols field.
		rows:   sd.Rows,   // Set the rows field to the value of the input's Rows field.
		values: sd.Values, // Set the values field to the value of the input's Values field.
	}

	// Return a pointer to the newly created Matrix object.
	return &m, nil
}
