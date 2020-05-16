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

// At returns the top level *Tensor at index k.
func (a *Tensor) At(k uint64) *Tensor {
	n := a.Shape[1:].Size()
	i := k * n
	return &Tensor{Data: a.Data[i : i+n], Shape: a.Shape[1:].Clone()}
}

// Clone makes and returns a copy of Tensor a
func (a *Tensor) Clone() *Tensor {
	var b Tensor
	b.Data = make([]float64, len(a.Data))
	copy(b.Data, a.Data)
	b.Shape = a.Shape.Clone()
	return &b
}

// Dot calculates and returns the Dot Product of Tensor a and b.
func (a *Tensor) Dot(b *Tensor) *Tensor {
	c := New(a.Shape[0], b.Shape[1])
	var n int
	var aR, bC, aC uint64
	for aR = 0; aR < a.Shape[0]; aR++ {
		for bC = 0; bC < b.Shape[1]; bC++ {
			for aC = 0; aC < a.Shape[1]; aC++ {
				i, j := aC+(a.Shape[1]*aR), bC+(b.Shape[1]*aC)
				c.Data[n] += a.Data[i] * b.Data[j]
			}
			n++
		}
	}
	return c
}

// Get returns a slice of Tensor a's elements.
func (a *Tensor) Get() []float64 {
	return a.Data
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

// Schur performs Schur product on Tensor a and b and stores the result in Tensor a.
func (a *Tensor) Schur(b *Tensor) {
	for k := range a.Data {
		a.Data[k] *= b.Data[k]
	}
}

// Set copies elements in data into Tensor a.
func (a *Tensor) Set(data []float64) {
	copy(a.Data, data)
}

// size calculates Tensor a's capacity from its dimensions.
func (a *Tensor) size() (size uint64) {
	return a.Shape.Size()
}

// Size returns the capacity of Tensor a.
func (a *Tensor) Size() uint64 {
	return uint64(cap(a.Data))
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

// Transpose flips Tensor a over its diagonal.
func (a *Tensor) Transpose() *Tensor {
	b := New(a.Shape...)
	b.Shape[0], b.Shape[1] = a.Shape[1], a.Shape[0]
	n := b.Shape[2:].Size()
	var i, j, k uint64
	for i = 0; i < b.Shape[0]; i++ {
		for j = 0; j < b.Shape[1]; j++ {
			m := (j * b.Shape[0]) + i
			copy(b.Data[k:k+n], a.Data[m:m+n])
			k++
		}
	}
	return b
}

// New creates and initializes a Tensor with n dimensions.
func New(n ...uint64) *Tensor {
	var a Tensor
	if len(n) < 2 {
		n = append(Shape{1}, n...)
	}
	a.Shape = append(Shape{}, n...)
	a.Data = make([]float64, a.size())
	return &a
}
