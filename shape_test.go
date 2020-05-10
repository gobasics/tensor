package tensor

import (
	"strconv"
	"testing"
)

func TestShapeSize(t *testing.T) {
	for k, v := range []struct {
		dimensions []int
		want       int
	}{
		{[]int{1, 1}, 1},
		{[]int{2, 2}, 4},
		{[]int{3, 3}, 9},
		{[]int{4, 4}, 16},
		{[]int{5, 5}, 25},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			tensor := New(v.dimensions...)
			got := tensor.Size()
			if v.want != got {
				t.Errorf("want %d, got %d", v.want, got)
			}
		})
	}
}
