package tensor

// Tensor is a datastructure that can store [1,n) dimensional float64 elements.
type Tensor struct {
	Data  []float64
	Shape Shape
}

// Add performs elementwise addition of Tensor delta to Tensor a.
// Tensor delta should be of the dimensions as Tensor a
func (a *Tensor) Add(delta *Tensor) {
	for k := range a.Data {
		a.Data[k] += delta.Data[k]
	}
}

// AddScalar adds delta to each element in Tensor a.
func (a *Tensor) AddScalar(delta float64) {
	for k := range a.Data {
		a.Data[k] += delta
	}
}

// At returns the top level *Tensor at index.
func (t *Tensor) At(index uint64) *Tensor {
	a := index * t.Shape[1:].Size()
	z := a + t.Shape[1:].Size()
	return &Tensor{Data: t.Data[a:z], Shape: t.Shape[1:]}
}

// Clone makes and returns a copy of Tensor a
func (a *Tensor) Clone() *Tensor {
	var b Tensor
	b.Data = make([]float64, len(a.Data))
	copy(b.Data, a.Data)
	b.Shape = a.Shape.Clone()
	return &b
}

// Set copies elements in data into Tensor a.
func (a *Tensor) Set(data []float64) {
	copy(a.Data, data)
}

// Dot calculates and returns the Dot Product of Tensor a and b.
func (a *Tensor) Dot(b *Tensor) (dot float64) {
	for k := range a.Data {
		dot += a.Data[k] * b.Data[k]
	}
	return dot
}

// SameShape returns true when Tensor a is of the same shape as Tensor b.
func (a *Tensor) SameShape(b *Tensor) bool {
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	for k := range a.Shape {
		if a.Shape[k] != b.Shape[k] {
			return false
		}
	}
	return true
}

// size calculates Tensor a's capacity from its dimensions.
func (a *Tensor) size() (size uint64) {
	return a.Shape.Size()
}

// Size returns the capacity of Tensor a.
func (a *Tensor) Size() uint64 {
	return uint64(cap(a.Data))
}

// Schur performs Schur product on Tensor a and b and stores the result in Tensor a.
func (a *Tensor) Schur(b *Tensor) {
	for k := range a.Data {
		a.Data[k] *= b.Data[k]
	}
}

// Square performs an elementwise squaring of Tensor a.
func (a *Tensor) Square() {
	for k := range a.Data {
		a.Data[k] *= a.Data[k]
	}
}

// Sub performs an elementwise subtraction of Tensor b from Tensor a.
func (a *Tensor) Sub(b *Tensor) {
	for k := range b.Data {
		a.Data[k] -= b.Data[k]
	}
}

// New creates and initializes a Tensor with n dimensions.
func New(n ...uint64) *Tensor {
	var a Tensor
	a.Shape = n
	a.Data = make([]float64, a.size())
	return &a
}
