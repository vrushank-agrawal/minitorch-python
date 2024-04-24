from numba import njit

import minitorch
from minitorch import fast_ops

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-M", action="store_true", default=False)
parser.add_argument("-Z", action="store_true", default=False)
parser.add_argument("-R", action="store_true", default=False)
parser.add_argument("-MM", action="store_true", default=False)
args = parser.parse_args()

# MAP
if args.M:
    print("MAP")
    tmap = minitorch.fast_ops.tensor_map(njit()(minitorch.operators.id))
    out, a = minitorch.zeros((10,)), minitorch.zeros((10,))
    tmap(*out.tuple(), *a.tuple())
    print(tmap.parallel_diagnostics(level=3))

# ZIP
if args.Z:
    print("ZIP")
    out, a, b = minitorch.zeros((10,)), minitorch.zeros((10,)), minitorch.zeros((10,))
    tzip = minitorch.fast_ops.tensor_zip(njit()(minitorch.operators.eq))

    tzip(*out.tuple(), *a.tuple(), *b.tuple())
    print(tzip.parallel_diagnostics(level=3))

# REDUCE
if args.R:
    print("REDUCE")
    out, a = minitorch.zeros((1,)), minitorch.zeros((10,))
    treduce = minitorch.fast_ops.tensor_reduce(njit()(minitorch.operators.add))

    treduce(*out.tuple(), *a.tuple(), 0)
    print(treduce.parallel_diagnostics(level=3))


# MM
if args.MM:
    print("MATRIX MULTIPLY")
    out, a, b = (
        minitorch.zeros((1, 10, 10)),
        minitorch.zeros((1, 10, 20)),
        minitorch.zeros((1, 20, 10)),
    )
    tmm = minitorch.fast_ops.tensor_matrix_multiply

    tmm(*out.tuple(), *a.tuple(), *b.tuple())
    print(tmm.parallel_diagnostics(level=3))

else:
    print("No tests")