package tensor

import (
	"strconv"
	"testing"
)

func TestShapeSize(t *testing.T) {
	for k, v := range []struct {
		dimensions []uint64
		want       uint64
	}{
		{[]uint64{1, 1}, 1},
		{[]uint64{2, 2}, 4},
		{[]uint64{3, 3}, 9},
		{[]uint64{4, 4}, 16},
		{[]uint64{5, 5}, 25},
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
