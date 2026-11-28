import pathlib
import numpy as np
import xarray as xr

import torch

from ..image.image import OmeTiffFile
from ..image.region import RegionSet

class BatchImageGenerator:

    def __init__(
        self,
        tif=OmeTiffFile,
        image=0,
        level=0
    ):
        self.tif = tif
        self.image = image
        self.level = level

    def _validate_image_size(self, image_size):

        if image_size is None:
            image_size = {"X": 256, "Y": 256}

        if not isinstance(image_size, dict) or not image_size:
            raise ValueError("image_size must be a non-empty dictionary.")

        named_shape = self.tif[self.image][self.level].named_shape

        for axis, size in image_size.items():

            if axis not in named_shape:
                raise ValueError(f"Unknown axis: {axis}")

            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"Invalid size for {axis}: {size}")

            if size > named_shape[axis]:
                raise ValueError(
                    f"{axis} size {size} exceeds image dimension "
                    f"{named_shape[axis]}"
                )

        return image_size, named_shape

    def generate_batch(
        self,
        batch_size=64,
        image_size=None,
        output_shape=("X", "Y", "S"),
    ):

        image_size, named_shape = self._validate_image_size(image_size)

        # Generate random valid coordinates
        coords = {
            axis: np.random.randint(
                0,
                named_shape[axis] - size + 1,
                size=batch_size,
            )
            for axis, size in image_size.items()
        }

        # Extract patches
        result = np.stack([
            self.tif.sel(
                image=self.image,
                level=self.level,
                output_shape=output_shape,
                **{
                    axis: (start, start + image_size[axis])
                    for axis, start in zip(image_size, starts)
                },
            ).to_numpy()
            for starts in zip(*coords.values())
        ])

        return torch.from_numpy(result)



class ROIBatchImageGenerator:

    def __init__(
        self, 
        tif: OmeTiffFile,
        rois: pathlib.Path | str | xr.Dataset,
        image = 0,
        level = 0,
    ):

        self.tif = tif
        self.image = image
        self.level = level

        self.rois = self._validate_rois(rois)

    def _validate_rois(self, rois):
        if isinstance(rois, str) | isinstance(rois, pathlib.Path):
            rois = xr.load_dataset(rois)
            
        assert len(rois) > 1, ValueError("Needs to have ROIs, none found.")

        return RegionSet(rois)

    def _validate_image_size(self, image_size):

        if image_size is None:
            image_size = {"X": 256, "Y": 256}

        if not isinstance(image_size, dict) or not image_size:
            raise ValueError("image_size must be a non-empty dictionary.")

        named_shape = self.tif[self.image][self.level].named_shape

        for axis, size in image_size.items():

            if axis not in named_shape:
                raise ValueError(f"Unknown axis: {axis}")

            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"Invalid size for {axis}: {size}")

            if size > named_shape[axis]:
                raise ValueError(
                    f"{axis} size {size} exceeds image dimension "
                    f"{named_shape[axis]}"
                )

        return image_size, named_shape

    def generate_batch(
        self,
        batch_size=64,
        image_size=None,
        output_shape=("X", "Y", "S"),
    ):
        image_size, named_shape = self._validate_image_size(image_size)

        # Sample coordinates once
        sampled = self.rois.sample_regions(n=batch_size).astype(int)

        # Setup patch coordinates, adjust so the sampled coords
        # are centered.
        coords = {
            axis: np.clip(
                sampled.sel(coordinate=axis).values - image_size[axis] // 2,
                0,
                named_shape[axis] - image_size[axis],
            ).astype(int)
            for axis in image_size
        }

        # Extract patches
        result = np.stack([
            self.tif.sel(
                image=self.image,
                level=self.level,
                output_shape=output_shape,
                **{
                    axis: (start, start + image_size[axis])
                    for axis, start in zip(image_size, starts)
                },
            ).to_numpy()
            for starts in zip(*coords.values())
        ])

        return torch.from_numpy(result)