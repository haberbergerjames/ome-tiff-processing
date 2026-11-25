import zarr
import tifffile
import xmltodict

# from typing import tuple



class OmeTiffFile:

    def __init__(self, filepath: str):
        self._filepath = filepath
        self.tif = tifffile.TiffFile(self._filepath)

        self.images = {
            series_id: OmeTiffImage(series_data)
            for series_id, series_data 
            in enumerate(self.tif.series)
        }


    def sel(
        self,
        image: int,
        level: int,
        xbounds: tuple[int, int],
        ybounds: tuple[int, int],
    ):
        img = self.images[image]
        arr = img.levels[level]

        x1, x2 = xbounds
        y1, y2 = ybounds

        axes = img._series.axes

        selection = [
            slice(None) if axis not in "YX" else
            slice(y1, y2) if axis == "Y" else
            slice(x1, x2)
            for axis in axes
        ]

        return arr[tuple(selection)]

class OmeTiffImage:

    def __init__(self, series: tifffile.TiffPageSeries):
        self._series = series
        self.store = zarr.open(
            self._series.levels[0].aszarr(), 
            mode="r"
        )

        # We need to handle both single level and multi-level
        # instances, we'll just handle them here as two explicit
        # cases.
        self.levels = {}

        if isinstance(self.store, zarr.Array):
            self.levels[0] = self.store

        elif isinstance(self.store, zarr.Group):
            for level_id, level_data in self.store.arrays():
                self.levels[int(level_id)] = level_data
