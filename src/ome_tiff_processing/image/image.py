import zarr
import pathlib
import tifffile
import xarray as xr

class OmeTiffFile:

    def __init__(self, filepath: str|pathlib.Path):
        # Simple check
        self._filepath = pathlib.Path(filepath)
        assert self._filepath.exists()

        # Load initial structure
        self.tif = tifffile.TiffFile(self._filepath)

        # Load each image
        self.images = {}
        self.load_images()
        
    def load_images(self):
        for image_idx, image in enumerate(self.tif.series):
            self.images[image_idx] = OmeTiffImage(image)
    
    def __repr__(self):
        return "\n\n".join([
            f"[Image {image_idx}]\n{repr(image)}"
            for image_idx, image in self.images.items()
        ])

    def __getitem__(self, index):
        return self.images[index]

    def sel(self, image=0, level=0, output_shape=None, **dimensions):
        return self.images[image].sel(
            level=level,
            output_shape=output_shape,
            **dimensions
        )




class OmeTiffImage:

    def __init__(self, image: tifffile.TiffPageSeries):
        # should capture information about:
        # top level dimension names, will be passed to lower ones
        self.image = image
        self.zarr = zarr.open(
            self.image.levels[0].aszarr(), 
            mode="r"
        )

        # Set the dimension names for the global object. helps with
        # Selecting and resolving between different layers.
        self.dimension_names = (
            self.zarr.metadata.dimension_names
            if isinstance(self.zarr, zarr.Array)
            else next(
                level_data
                for level_id, level_data in sorted(
                    self.zarr.arrays(), key=lambda x: int(x[0])
                )
                if int(level_id) == 0
            ).metadata.dimension_names
        )

        # We need to handle both single level and multi-level
        # instances, we'll just handle them here as two explicit
        # cases.
        self.levels = {}
        if isinstance(self.zarr, zarr.Array):
            self.levels[0] = OmeTiffLevel(self.store)

        elif isinstance(self.zarr, zarr.Group):
            for level_id, level_data in sorted(self.zarr.arrays(), key=lambda x: int(x[0])):
                self.levels[int(level_id)] = OmeTiffLevel(level_data, self)
        
    def __repr__(self):
        return "\n".join([
            f"|-> [Level {level_idx}] {repr(level)}" 
            for level_idx, level in self.levels.items()
        ])
            
    def __getitem__(self, index):
        return self.levels[index]
    
    def sel(self, level=0, output_shape=None, **dimensions):
        return self.levels[level].sel(
            output_shape=output_shape,
            **dimensions
        )



class OmeTiffLevel:

    def __init__(self, level: zarr.core.array.Array, parent: OmeTiffImage):
        # Reference to parent, useful when resolving indexing.
        self._parent = parent
        self.zarr = level

        # for resolving dimension names between the two
        self._dimension_name_mapping = {
            k: v for k, v in 
            zip(
                self._parent.dimension_names,
                self.zarr.metadata.dimension_names
            )
        }

    def __repr__(self):
        shape = self.zarr.shape
        names = self.zarr.metadata.dimension_names
        return f"shape={shape}, names={names}"

    def resolve_sel_indexing(self, dimensions):
        selection = [slice(None)] * self.zarr.ndim

        for dim, bounds in dimensions.items():
            actual_name = self._dimension_name_mapping[dim]
            idx = self.zarr.metadata.dimension_names.index(actual_name)

            if bounds is not None:
                selection[idx] = slice(*bounds)

        return tuple(selection)

    def sel(self, output_shape=None, **dimensions):
        selection = self.resolve_sel_indexing(dimensions)
        array = self.zarr[selection]

        if output_shape is not None:
            actual_axes = [
                self._dimension_name_mapping[dim]
                for dim in output_shape
            ]

            transpose_order = [
                self.zarr.metadata.dimension_names.index(dim)
                for dim in actual_axes
            ]

            array = array.transpose(transpose_order)

        array = xr.DataArray(array, dims=output_shape)

        return array