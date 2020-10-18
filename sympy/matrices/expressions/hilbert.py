from sympy.core.sympify import _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy import S, factorial, Rational, Product, Dummy, binomial

class HilbertMatrix(MatrixExpr):
    r""" Generates a Hilbert matrix.

        Parameters
        ==========

        n : integer
        Size of the Hilbert matrix.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Hilbert_matrix

    """
    def __new__(cls, n):
        n = _sympify(n)
        cls._check_dim(n)

        obj = super().__new__(cls, n)
        return obj

    n = property(lambda self: self.args[0])  # type: ignore
    shape = property(lambda self: (self.n, self.n))  # type: ignore

    def _entry(self, i, j, **kwargs):
        return S.One/(i + j + 1)

    def _eval_determinant(self):
        d = Dummy('i')
        cn = Product(factorial(d), (d, 1, self.n-1))
        c2n = Product(factorial(d), (d, 1, 2*self.n-1))
        return cn**4/c2n

    def _eval_inverse(self):
        return InverseHilbertMatrix(self.n)


class InverseHilbertMatrix(HilbertMatrix):
    """ Class for representing the inverse of a Hilbert matrix """

    def _entry(self, i, j, **kwargs):
        return ((-1)**(i+j)*(i+j+1)*binomial(self.n + i, self.n - j - 1)*
                binomial(self.n + j, self.n - i - 1)*binomial(i+j, i)**2)

    def _eval_inverse(self):
        return HilbertMatrix(self.n)

    def _eval_determinant(self):
        d = Dummy('i')
        cn = Product(factorial(d), (d, 1, self.n-1))
        c2n = Product(factorial(d), (d, 1, 2*self.n-1))
        return c2n/cn**4
