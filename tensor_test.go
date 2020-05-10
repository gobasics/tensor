package tensor

import (
	"fmt"
	"strconv"
	"testing"
)

func TestAdd(t *testing.T) {
	for k, v := range []struct {
		data  []float64
		delta []float64
		want  []float64
	}{
		{[]float64{1, 2, 3}, []float64{2, 3, 4}, []float64{3, 5, 7}},
		{[]float64{1, 2, 3}, []float64{3, 4, 5}, []float64{4, 6, 8}},
		{[]float64{1, 2, 3}, []float64{4, 5, 6}, []float64{5, 7, 9}},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			a := New(3)
			a.Set(v.data)
			b := New(3)
			b.Set(v.delta)
			a.Add(b)
			want := fmt.Sprintf("%v", v.want)
			got := fmt.Sprintf("%v", a.data)
			if want != got {
				t.Errorf("want %s, got %s", want, got)
			}
		})
	}
}

func TestAddScalar(t *testing.T) {
	for k, v := range []struct {
		data  []float64
		delta float64
		want  []float64
	}{
		{[]float64{1, 2, 3}, 1, []float64{2, 3, 4}},
		{[]float64{1, 2, 3}, 2, []float64{3, 4, 5}},
		{[]float64{1, 2, 3}, 3, []float64{4, 5, 6}},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			tensor := New(3)
			tensor.data = v.data
			tensor.AddScalar(v.delta)
			want := fmt.Sprintf("%v", v.want)
			got := fmt.Sprintf("%v", tensor.data)
			if want != got {
				t.Errorf("want %s, got %s", want, got)
			}
		})
	}
}

func TestDot(t *testing.T) {
	for k, v := range []struct {
		a    []float64
		b    []float64
		want float64
	}{
		{[]float64{1, 2, 3}, []float64{2, 3, 4}, 20},
		{[]float64{1, 2, 3}, []float64{3, 4, 5}, 26},
		{[]float64{1, 2, 3}, []float64{4, 5, 6}, 32},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			a := New(3)
			a.Set(v.a)

			b := New(3)
			b.Set(v.b)

			got := a.Dot(b)
			if v.want != got {
				t.Errorf("want %g, got %g", v.want, got)
			}
		})
	}
}

func TestSize(t *testing.T) {
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
