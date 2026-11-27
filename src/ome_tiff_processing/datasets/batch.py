import torch
import numpy as np

from ..image.image import OmeTiffFile

class BatchImageGenerator:

    def __init__(
        self, 
        tif = OmeTiffFile,
        image = 0,
        level = 0
    ):

        self.tif = tif
        self.image = image
        self.level = level


    def generate_batch(
        self,
        batch_size=64,
        image_size=None,
        output_shape=("X", "Y", "S"),
    ):
        if image_size is None:
            image_size = {"X": 256, "Y": 256}

        # Get underlying image dimensions
        named_shape = self.tif[self.image][self.level].named_shape

        # Validate requested dimensions
        for axis, size in image_size.items():
            if axis not in named_shape:
                raise ValueError(f"Unknown axis: {axis}")

            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"Invalid size for {axis}: {size}")

            if size > named_shape[axis]:
                raise ValueError(
                    f"{axis} size {size} exceeds image dimension "
                    f"{named_shape[axis]}"
                )

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
