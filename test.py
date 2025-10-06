"""
This script demonstrates zero-copy interop between xarray and pyarrow,
via NumPy. It uses NumPy's FixedShapeTensorArray type to create a
pyarrow Table with a column that's ultimately backed by NumPy memory.
"""
# /// script
# dependencies = [
#   "pyarrow",
#   "numpy",
#   "xarray",
# ]
# ///


import pyarrow as pa
import numpy as np
import pyarrow.parquet as pq
import xarray as xr


ROWS, BANDS, Y, X = 50, 8, 512, 512

arr = np.random.randint(0, 254, (ROWS, BANDS, Y, X), dtype=np.uint8)

# Zero copy from numpy to Arrow
fst = pa.FixedShapeTensorArray.from_numpy_ndarray(arr)

# Create a Table with a FixedShapeTensor column
t = pa.Table.from_arrays(
    [
        pa.array(np.arange(len(arr))),
        fst,
    ],
    names=["id", "arr"],
)

print("Schema:")
print(t.schema)

# Zero copy from Arrow to xarray (via NumPy)
# Only works by chunks of a ChunkedArray
da = xr.DataArray(t[1].chunks[0].to_numpy_ndarray())

# Prove that it's zero copy:
assert da[0, 0, 0, 0] != 42
arr[0, 0, 0, 0] = 42
assert da[0, 0, 0, 0] == 42


# pq.write_table(t, "examples/fixed-shape-tensor.parquet")
