package tensor

// Shape holds dimensions of a Tensor
type Shape []int

// Clone makes and returns a copy of Shape a.
func (a Shape) Clone() Shape {
	b := make(Shape, len(a))
	copy(b, a)
	return b
}

// Size calculates the Tensor capacity of Shape a
func (a Shape) Size() (size int) {
	size = 1
	for _, n := range a {
		size *= n
	}
	return size
}
