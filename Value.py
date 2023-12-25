import math

class Value:

    def __init__(self, data, _children=(), _op="", label="") -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other) -> int:
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad

        output._backward = _backward

        return output
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other) -> int:
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward

        return output
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "The Value classo only supports ints or floats for now!"
        output = Value(self.data**other, (self, ), f"**(other)")

        def _backward():
            self.grad += (other * self.data**(other-1)) * output.grad
        output._backward = _backward

        return output

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        output = Value(t, (self, ), "tanh")

        def _backward():
            self.grad += (1 - t**2) * output.grad

        output._backward = _backward

        return output
    
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def exp(self):
        x = self.data
        output = Value(math.exp(x), (self, ), "exp")

        def _backward():
            self.grad += output.data * output.grad
        output._backward = _backward

        return output