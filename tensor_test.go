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
			got := fmt.Sprintf("%v", a.Get())
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
			tensor.Data = v.data
			tensor.AddScalar(v.delta)
			want := fmt.Sprintf("%v", v.want)
			got := fmt.Sprintf("%v", tensor.Get())
			if want != got {
				t.Errorf("want %s, got %s", want, got)
			}
		})
	}
}

func TestAt(t *testing.T) {
	for k, v := range []struct {
		at     uint64
		tensor *Tensor
		want   *Tensor
	}{
		{
			0,
			&Tensor{Data: []float64{5, 4, 2, 6, 7, 0, 3, 1}, Shape: Shape{2, 2, 2}},
			&Tensor{Data: []float64{5, 4, 2, 6}, Shape: Shape{2, 2}},
		},
		{
			1,
			&Tensor{Data: []float64{5, 4, 2, 6, 7, 0, 3, 1}, Shape: Shape{2, 2, 2}},
			&Tensor{Data: []float64{7, 0, 3, 1}, Shape: Shape{2, 2}},
		},
		{
			1,
			&Tensor{Data: []float64{5, 4, 2, 6, 7, 0, 3, 1}, Shape: Shape{2, 4}},
			&Tensor{Data: []float64{7, 0, 3, 1}, Shape: Shape{1, 4}},
		},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			want := fmt.Sprintf("%v", v.want)
			got := fmt.Sprintf("%v", v.tensor.At(v.at))
			if want != got {
				t.Errorf("want %s, got %s", want, got)
			}
		})
	}
}

func TestClone(t *testing.T) {
	for k, v := range []struct {
		a    *Tensor
		want *Tensor
	}{
		{&Tensor{Data: []float64{1, 2, 3, 4}, Shape: Shape{2, 2}}, &Tensor{Data: []float64{1, 2, 3, 4}, Shape: Shape{2, 2}}},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			want := fmt.Sprintf("%+v", v.want)
			got := fmt.Sprintf("%+v", v.a.Clone())
			if want != got {
				t.Errorf("want %s, got %s", want, got)
			}
		})
	}
}

func TestDot(t *testing.T) {
	for k, v := range []struct {
		a    *Tensor
		b    *Tensor
		want *Tensor
	}{
		{
			&Tensor{[]float64{1, 2, 3, 4, 5, 6}, Shape{2, 3}},
			&Tensor{[]float64{7, 8, 9, 10, 11, 12}, Shape{3, 2}},
			&Tensor{[]float64{58, 64, 139, 154}, Shape{2, 2}},
		},
		{
			&Tensor{[]float64{1, 2, 3}, Shape{1, 3}},
			&Tensor{[]float64{2, 3, 4}, Shape{3, 1}},
			&Tensor{[]float64{20}, Shape{1, 1}},
		},
		{
			&Tensor{[]float64{1, 2, 3}, Shape{1, 3}},
			&Tensor{[]float64{2, 3, 4, 5, 6, 7}, Shape{3, 2}},
			&Tensor{[]float64{28, 34}, Shape{1, 2}},
		},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			want := fmt.Sprintf("%+v", v.want)
			got := fmt.Sprintf("%+v", v.a.Dot(v.b))
			if want != got {
				t.Errorf("want %s, got %s", want, got)
			}
		})
	}
}

func TestSameShape(t *testing.T) {
	for k, v := range []struct {
		a    *Tensor
		b    *Tensor
		want bool
	}{
		{New(1, 2, 3), New(1, 2, 3), true},
		{New(1, 2, 2), New(1, 2, 3), false},
		{New(1, 2, 2, 3), New(1, 2, 3), false},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			got := v.a.SameShape(v.b)
			if v.want != got {
				t.Errorf("want %t, got %t", v.want, got)
			}
		})
	}
}

func TestSize(t *testing.T) {
	for k, v := range []struct {
		dimensions []uint64
		want       uint64
	}{
		{[]uint64{}, 1},
		{[]uint64{1}, 1},
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

func TestSchur(t *testing.T) {
	for k, v := range []struct {
		a    *Tensor
		b    *Tensor
		want *Tensor
	}{
		{
			&Tensor{[]float64{1, 2, 3, 4}, Shape{2, 2}},
			&Tensor{[]float64{2, 3, 4, 5}, Shape{2, 2}},
			&Tensor{[]float64{2, 6, 12, 20}, Shape{2, 2}},
		},
		{
			&Tensor{[]float64{2, 3, 4, 5}, Shape{2, 2}},
			&Tensor{[]float64{3, 4, 5, 6}, Shape{2, 2}},
			&Tensor{[]float64{6, 12, 20, 30}, Shape{2, 2}},
		},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			v.a.Schur(v.b)
			want := fmt.Sprintf("%+v", v.want.Get())
			got := fmt.Sprintf("%+v", v.a.Get())
			if want != got {
				t.Errorf("want %s, got %s", want, got)
			}
		})
	}
}

func TestSquare(t *testing.T) {
	for k, v := range []struct {
		a    []float64
		want []float64
	}{
		{[]float64{1, 2, 3}, []float64{1, 4, 9}},
		{[]float64{2, 3, 4}, []float64{4, 9, 16}},
		{[]float64{3, 4, 5}, []float64{9, 16, 25}},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			a := New(3)
			a.Set(v.a)
			a.Square()
			want := fmt.Sprintf("%+v", v.want)
			got := fmt.Sprintf("%+v", a.Get())
			if want != got {
				t.Errorf("want %s, got %s", want, got)
			}
		})
	}
}

func TestSub(t *testing.T) {
	for k, v := range []struct {
		data  []float64
		delta []float64
		want  []float64
	}{
		{[]float64{1, 2, 3}, []float64{2, 3, 4}, []float64{-1, -1, -1}},
		{[]float64{1, 2, 3}, []float64{3, 4, 5}, []float64{-2, -2, -2}},
		{[]float64{1, 2, 3}, []float64{4, 5, 6}, []float64{-3, -3, -3}},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			a := New(3)
			a.Set(v.data)
			b := New(3)
			b.Set(v.delta)
			a.Sub(b)
			want := fmt.Sprintf("%v", v.want)
			got := fmt.Sprintf("%v", a.Get())
			if want != got {
				t.Errorf("want %s, got %s", want, got)
			}
		})
	}
}

func TestTranspose(t *testing.T) {
	for k, v := range []struct {
		a, want *Tensor
	}{
		{
			&Tensor{[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, Shape{3, 4}},
			&Tensor{[]float64{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}, Shape{4, 3}}},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			want := fmt.Sprintf("%+v", v.want)
			got := fmt.Sprintf("%+v", v.a.Transpose())
			if want != got {
				t.Errorf("want %s, got %s", want, got)
			}
		})
	}
}
