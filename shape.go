package tensor

// Shape holds dimensions of a Tensor
type Shape []uint64

// Clone makes and returns a copy of Shape a.
func (a Shape) Clone() Shape {
	b := make(Shape, len(a))
	copy(b, a)
	return b
}

// Size calculates the Tensor capacity of Shape a
func (a Shape) Size() (size uint64) {
	size = 1
	for _, n := range a {
		size *= uint64(n)
	}
	return size
}
