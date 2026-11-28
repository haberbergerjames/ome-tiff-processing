import shapely
import numpy as np
import xarray as xr

class Region:

    def __init__(self, roi: xr.DataArray):
        self.roi = roi
        self.polygon = shapely.Polygon(roi.as_numpy())

    def sample_polygon(self, n=1, rng=None):

        if rng is None:
            rng = np.random.default_rng()

        if self.polygon.is_empty or not self.polygon.is_valid:
            raise ValueError("Polygon must be non-empty and valid.")

        triangles = list(
            shapely.constrained_delaunay_triangles(self.polygon).geoms
        )

        areas = np.array([t.area for t in triangles])
        probabilities = areas / areas.sum()

        # Select triangles proportional to area
        indices = rng.choice(
            len(triangles),
            size=n,
            p=probabilities
        )

        # Sample uniformly inside selected triangles
        points = []

        for idx in indices:

            coords = np.asarray(triangles[idx].exterior.coords[:3])

            u, v = rng.random(2)

            # Uniform barycentric sampling
            if u + v > 1:
                u, v = 1 - u, 1 - v

            point = coords[0] + u * (coords[1] - coords[0]) \
                            + v * (coords[2] - coords[0])

            points.append(point)

        return np.asarray(points)
    
    def __repr__(self):
        area = round(
            self.polygon.area,
            3
        )
        coords = tuple(
            round(x, 3)
            for x 
            in self.polygon.centroid.coords[0]
        )

        return f"centroid: {coords}; area: {area}"

class RegionSet:

    def __init__(self, regions_of_interest: xr.Dataset):
        self.rois = regions_of_interest
        self.roi_set = {
            region_name: Region(regions_of_interest[region_name])
            for region_name 
            in sorted(list(regions_of_interest.data_vars))
        }
    
    def normalized_area(self):
    
        total_size = sum([
            self.roi_set[k].polygon.area
            for k in self.roi_set 
        ])

        return {
            k: v.polygon.area / total_size
            for k, v
            in self.roi_set.items()
        }
    
    def __getitem__(self, key):
        return self.roi_set[key]

    def __repr__(self):
        return "\n".join([
            f"[{k}]\n|-> {repr(self.roi_set[k])}"
            for k in self.roi_set
        ]) 
    
    def sample_regions(self, pvals=None, n=1000):
        # if pvals aren't provided, just sample based on relative area.
        if pvals == None:
            pvals = list(self.normalized_area().values())

        # multinomial sampling of the number of points in each region
        region_sampling = np.random.multinomial(
            n=n,
            pvals=pvals
        )

        # uniform polygonal sampling using delauney triangulation
        arr = np.concat([
            self[roi].sample_polygon(n=n)
            for roi, n
            in list(zip(self.roi_set.keys(), region_sampling))
        ])

        coordinates = self.rois.coords["coordinate"].values

        return xr.DataArray(
            arr,
            dims=("sample", "coordinate"),
            coords={
                "coordinate": xr.IndexVariable("coordinate", coordinates)
            }
        )