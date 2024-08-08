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

// NewMatrix create a new Matrix
func NewMatrix(cols, rows uint32) *Matrix {
	m := Matrix{
		cols:   cols,
		rows:   rows,
		values: make([]float64, cols*rows),
	}
	return &m
}

func NewFromSlice(slc []float64) *Matrix {
	m := NewMatrix(uint32(len(slc)), 1)
	m.values = slc
	return m
}

// ApplyFunction appies a function to the matrix elements
func (m *Matrix) ApplyFunction(f NeuralFunction) *Matrix {
	o := NewMatrix(m.cols, m.rows)
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			mC, _ := m.At(x, y)
			o.Set(x, y, f(mC))
		}
	}
	return o
}

// Cols returns the number of columns in the matrix
func (m *Matrix) Cols() uint32 {
	return m.cols
}

// Rows returns the number of rows in the matrix
func (m *Matrix) Rows() uint32 {
	return m.rows
}

// Values returns a slice of all the values in the matix
func (m *Matrix) Values() []float64 {
	return m.values
}

// At returns the value of the matrix at the specified column and row
func (m *Matrix) At(col, row uint32) (float64, error) {
	if col >= m.cols {
		return 0, fmt.Errorf("column out of range: %v maximum, %v requested", m.cols-1, col)
	}
	if row >= m.rows {
		return 0, fmt.Errorf("row out of range: %v maximum, %v requested", m.rows-1, row)
	}
	return m.values[row*m.cols+col], nil
}

// Set assigns a value to the specified cell at column, row
func (m *Matrix) Set(col, row uint32, v float64) error {
	if col >= m.cols {
		return fmt.Errorf("column out of range: %v maximum, %v requested", m.cols-1, col)
	}
	if row >= m.rows {
		return fmt.Errorf("row out of range: %v maximum, %v requested", m.rows-1, row)
	}
	m.values[row*m.cols+col] = v
	return nil
}

// Multiply multiplies the matrix with a target matrix, returning a new matrix
func (m *Matrix) Multiply(tgt *Matrix) (*Matrix, error) {
	if m.cols != tgt.rows {
		return nil, errors.New("shape error")
	}
	o := NewMatrix(tgt.cols, m.rows)
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			var v float64
			for k := uint32(0); k < m.cols; k++ {
				mC, _ := m.At(k, y)
				tC, _ := tgt.At(x, k)
				v += mC * tC
			}
			o.Set(x, y, v)
		}
	}
	return o, nil
}

// MultiplyScalar multiplies the matrix with a single float, returning a new matrix
func (m *Matrix) MultiplyScalar(v float64) *Matrix {
	o := NewMatrix(m.cols, m.rows)
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			mC, _ := m.At(x, y)
			o.Set(x, y, v*mC)
		}
	}
	return o
}

func (m *Matrix) MultiplyElements(tgt *Matrix) (*Matrix, error) {
	if m.cols != tgt.cols || m.rows != tgt.rows {
		return nil, errors.New("shape error")
	}
	o := NewMatrix(m.cols, m.rows)
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			mC, _ := m.At(x, y)
			tC, _ := tgt.At(x, y)
			o.Set(x, y, mC*tC)
		}
	}
	return o, nil
}

// Add adds the matrix with a target matrix, returning a new matrix
func (m *Matrix) Add(tgt *Matrix) (*Matrix, error) {
	if m.cols != tgt.cols || m.rows != tgt.rows {
		return nil, errors.New("shape error")
	}
	o := NewMatrix(m.cols, m.rows)
	for y := uint32(0); y < m.rows; y++ {
		for x := uint32(0); x < m.cols; x++ {
			mC, _ := m.At(x, y)
			tC, _ := tgt.At(x, y)
			o.Set(x, y, mC+tC)
		}
	}
	return o, nil
}

// Add adds the matrix with a float value, returning a new matrix
func (m *Matrix) AddScalar(v float64) *Matrix {
	o := NewMatrix(m.cols, m.rows)
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			mC, _ := m.At(x, y)
			o.Set(x, y, v+mC)
		}
	}
	return o
}

// Negative returns the negative of the matrix
func (m *Matrix) Negative() *Matrix {
	o := NewMatrix(m.cols, m.rows)
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			mC, _ := m.At(x, y)
			o.Set(x, y, -mC)
		}
	}
	return o
}

// Transpose returns the transpose of the matrix
func (m *Matrix) Transpose() *Matrix {
	o := NewMatrix(m.rows, m.cols)
	for y := uint32(0); y < o.rows; y++ {
		for x := uint32(0); x < o.cols; x++ {
			mC, _ := m.At(y, x)
			o.Set(x, y, mC)
		}
	}
	return o
}

func (m *Matrix) SetValues(vals []float64) error {
	if vals == nil {
		return errors.New("missing values")
	}
	if len(vals) != len(m.values) {
		return errors.New("size error")
	}
	m.values = vals
	return nil
}

func (m *Matrix) ToSaveData() *MatrixSaveData {
	sd := MatrixSaveData{
		Cols:   m.cols,
		Rows:   m.rows,
		Values: m.values,
	}
	return &sd
}

func MatrixFromSaveData(sd *MatrixSaveData) (*Matrix, error) {
	if sd == nil {
		return nil, errors.New("missing save data")
	}

	m := Matrix{
		cols:   sd.Cols,
		rows:   sd.Rows,
		values: sd.Values,
	}

	return &m, nil
}
