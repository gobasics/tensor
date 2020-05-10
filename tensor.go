package tensor

// Tensor is a datastructure that can store [1,n) dimensional float64 elements.
type Tensor struct {
	data       []float64
	dimensions []int
}

// AddScalar adds delta to each element in Tensor a.
func (a *Tensor) AddScalar(delta float64) {
	for k := range a.data {
		a.data[k] += delta
	}
}

// Add performs elementwise addition of Tensor delta to Tensor a.
// Tensor delta should be of the dimensions as Tensor a
func (a *Tensor) Add(delta *Tensor) {
	for k := range a.data {
		a.data[k] += delta.data[k]
	}
}

// Set copies elements in data into Tensor a.
func (a *Tensor) Set(data []float64) {
	copy(a.data, data)
}

// Dot calculates and returns the Dot Product of Tensor a and b.
func (a *Tensor) Dot(b *Tensor) (dot float64) {
	for k := range a.data {
		dot += a.data[k] * b.data[k]
	}
	return dot
}

// size calculates Tensor a's capacity from its dimensions.
func (a *Tensor) size() (size int) {
	size = 1
	for _, n := range a.dimensions {
		size *= n
	}
	return size
}

// Size returns the capacity of Tensor a.
func (a *Tensor) Size() int {
	return cap(a.data)
}

// New creates and initializes a Tensor with n dimensions.
func New(n ...int) *Tensor {
	var a Tensor
	a.dimensions = n
	a.data = make([]float64, a.size())
	return &a
}
