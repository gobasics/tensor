package tensor

// Tensor is a datastructure that can store [1,n) dimensional float64 elements.
type Tensor struct {
	data       []float64
	dimensions []int
}

// Add adds delta to each element in Tensor t.
func (t *Tensor) Add(delta float64) {
	for k := range t.data {
		t.data[k] += delta
	}
}

// Set copies elements in data into Tensor t.
func (t *Tensor) Set(data []float64) {
	copy(t.data, data)
}

// Dot calculates and returns the Dot Product of Tensor t and c.
func (t *Tensor) Dot(c *Tensor) (dot float64) {
	for k := range t.data {
		dot += t.data[k] * c.data[k]
	}
	return dot
}

// Calculates Tensor t's capacity from its dimensions.
func (t *Tensor) size() (size int) {
	size = 1
	for _, n := range t.dimensions {
		size *= n
	}
	return size
}

// size returns the capacity of Tensor t.
func (t *Tensor) Size() int {
	return cap(t.data)
}

// New creates and initializes a Tensor of dimensions.
func New(dimensions ...int) *Tensor {
	var t Tensor
	t.dimensions = dimensions
	t.data = make([]float64, t.size())
	return &t
}
