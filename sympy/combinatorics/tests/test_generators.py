from sympy.combinatorics.generators import symmetric, cyclic, alternating, \
    dihedral, rubik
from sympy.combinatorics.permutations import Permutation
from sympy.testing.pytest import raises

def test_generators():

    assert list(cyclic(6)) == [
        Permutation([0, 1, 2, 3, 4, 5]),
        Permutation([1, 2, 3, 4, 5, 0]),
        Permutation([2, 3, 4, 5, 0, 1]),
        Permutation([3, 4, 5, 0, 1, 2]),
        Permutation([4, 5, 0, 1, 2, 3]),
        Permutation([5, 0, 1, 2, 3, 4])]

    assert list(cyclic(10)) == [
        Permutation([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        Permutation([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
        Permutation([2, 3, 4, 5, 6, 7, 8, 9, 0, 1]),
        Permutation([3, 4, 5, 6, 7, 8, 9, 0, 1, 2]),
        Permutation([4, 5, 6, 7, 8, 9, 0, 1, 2, 3]),
        Permutation([5, 6, 7, 8, 9, 0, 1, 2, 3, 4]),
        Permutation([6, 7, 8, 9, 0, 1, 2, 3, 4, 5]),
        Permutation([7, 8, 9, 0, 1, 2, 3, 4, 5, 6]),
        Permutation([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]),
        Permutation([9, 0, 1, 2, 3, 4, 5, 6, 7, 8])]

    assert list(alternating(4)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([0, 2, 3, 1]),
        Permutation([0, 3, 1, 2]),
        Permutation([1, 0, 3, 2]),
        Permutation([1, 2, 0, 3]),
        Permutation([1, 3, 2, 0]),
        Permutation([2, 0, 1, 3]),
        Permutation([2, 1, 3, 0]),
        Permutation([2, 3, 0, 1]),
        Permutation([3, 0, 2, 1]),
        Permutation([3, 1, 0, 2]),
        Permutation([3, 2, 1, 0])]

    assert list(symmetric(3)) == [
        Permutation([0, 1, 2]),
        Permutation([0, 2, 1]),
        Permutation([1, 0, 2]),
        Permutation([1, 2, 0]),
        Permutation([2, 0, 1]),
        Permutation([2, 1, 0])]

    assert list(symmetric(4)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([0, 1, 3, 2]),
        Permutation([0, 2, 1, 3]),
        Permutation([0, 2, 3, 1]),
        Permutation([0, 3, 1, 2]),
        Permutation([0, 3, 2, 1]),
        Permutation([1, 0, 2, 3]),
        Permutation([1, 0, 3, 2]),
        Permutation([1, 2, 0, 3]),
        Permutation([1, 2, 3, 0]),
        Permutation([1, 3, 0, 2]),
        Permutation([1, 3, 2, 0]),
        Permutation([2, 0, 1, 3]),
        Permutation([2, 0, 3, 1]),
        Permutation([2, 1, 0, 3]),
        Permutation([2, 1, 3, 0]),
        Permutation([2, 3, 0, 1]),
        Permutation([2, 3, 1, 0]),
        Permutation([3, 0, 1, 2]),
        Permutation([3, 0, 2, 1]),
        Permutation([3, 1, 0, 2]),
        Permutation([3, 1, 2, 0]),
        Permutation([3, 2, 0, 1]),
        Permutation([3, 2, 1, 0])]

    assert list(dihedral(1)) == [
        Permutation([0, 1]), Permutation([1, 0])]

    assert list(dihedral(2)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([1, 0, 3, 2]),
        Permutation([2, 3, 0, 1]),
        Permutation([3, 2, 1, 0])]

    assert list(dihedral(3)) == [
        Permutation([0, 1, 2]),
        Permutation([2, 1, 0]),
        Permutation([1, 2, 0]),
        Permutation([0, 2, 1]),
        Permutation([2, 0, 1]),
        Permutation([1, 0, 2])]

    assert list(dihedral(5)) == [
        Permutation([0, 1, 2, 3, 4]),
        Permutation([4, 3, 2, 1, 0]),
        Permutation([1, 2, 3, 4, 0]),
        Permutation([0, 4, 3, 2, 1]),
        Permutation([2, 3, 4, 0, 1]),
        Permutation([1, 0, 4, 3, 2]),
        Permutation([3, 4, 0, 1, 2]),
        Permutation([2, 1, 0, 4, 3]),
        Permutation([4, 0, 1, 2, 3]),
        Permutation([3, 2, 1, 0, 4])]

    raises(ValueError, lambda: rubik(1))

    assert rubik(2) == [
        Permutation(23)(2, 19, 21, 8)(3, 17, 20, 10)(4, 6, 7, 5),
        Permutation(1, 5, 21, 14)(3, 7, 23, 12)(8, 10, 11, 9),
        Permutation(6, 18, 14, 10)(7, 19, 15, 11)(20, 22, 23, 21)]

    assert rubik(3) == [
        Permutation(53)(6, 44, 47, 18)(7, 41, 46, 21)(8, 38, 45, 24)(9, 15, 17, 11)(10, 12, 16, 14),
        Permutation(53)(3, 43, 50, 19)(4, 40, 49, 22)(5, 37, 48, 25),
        Permutation(2, 11, 47, 33)(5, 14, 50, 30)(8, 17, 53, 27)(18, 24, 26, 20)(19, 21, 25, 23),
        Permutation(53)(1, 10, 46, 34)(4, 13, 49, 31)(7, 16, 52, 28),
        Permutation(15, 42, 33, 24)(16, 43, 34, 25)(17, 44, 35, 26)(45, 51, 53, 47)(46, 48, 52, 50),
        Permutation(53)(12, 39, 30, 21)(13, 40, 31, 22)(14, 41, 32, 23)]
