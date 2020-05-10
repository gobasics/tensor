package tensor

// Shape holds dimensions of a Tensor
type Shape []int

// Size calculates the Tensor capacity of Shape s
func (s Shape) Size() (size int) {
	size = 1
	for _, n := range s {
		size *= n
	}
	return size
}
