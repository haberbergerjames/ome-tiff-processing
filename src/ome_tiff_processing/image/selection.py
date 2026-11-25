import re
import webbrowser
import threading
from queue import Queue

import numpy as np
import xarray as xr
import plotly.express as px

from dash import Dash, dcc, html, Input, Output, State


class ROISelectionTool:

    def __init__(self, image):
        self.image = image

        self.roi_results = Queue()
        self.app = None
        self._server_thread = None

        self.reference_level = None
        self.target_level = None

    def generate_dash_app(self, image):

        fig = px.imshow(image)

        fig.update_layout(
            dragmode="drawclosedpath",
            newshape=dict(
                line_color="red",
                fillcolor="rgba(255, 0, 0, 0.25)"
            )
        )

        app = Dash(__name__)

        app.layout = html.Div([
            dcc.Graph(
                id="image",
                figure=fig,
                config={
                    "modeBarButtonsToAdd": [
                        "drawclosedpath",
                        "eraseshape",
                    ]
                }
            ),

            html.Button("Export ROIs", id="export"),

            dcc.Store(
                id="roi-store",
                data=[]
            ),

            html.Div(id="status")
        ])

        @app.callback(
            Output("roi-store", "data"),
            Input("image", "relayoutData"),
            State("roi-store", "data"),
            prevent_initial_call=True
        )
        def update_rois(relayout_data, current_rois):

            if not relayout_data:
                return current_rois

            if "shapes" in relayout_data:
                return [
                    shape["path"]
                    for shape in relayout_data["shapes"]
                    if shape.get("type") == "path"
                ]

            return current_rois

        @app.callback(
            Output("status", "children"),
            Input("export", "n_clicks"),
            State("roi-store", "data"),
            prevent_initial_call=True
        )
        def export_rois(n_clicks, polygons):

            self.roi_results.put(polygons)

            return f"Exported {len(polygons)} regions"

        return app

    def serve(self, reference_level=3, port=8050):

        self.reference_level = reference_level

        image = self.image.sel(
            level=reference_level,
            output_shape=("Y", "X", "S")
        )

        self.app = self.generate_dash_app(np.asarray(image))

        self._server_thread = threading.Thread(
            target=lambda: self.app.run(
                host="127.0.0.1",
                port=port,
                debug=False,
                use_reloader=False
            ),
            daemon=True
        )

        self._server_thread.start()

        url = f"http://127.0.0.1:{port}"
        print(f"ROI selector running at {url}")

        webbrowser.open(url)

    @staticmethod
    def parse_polygon(path):

        coordinates = re.findall(
            r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?",
            path
        )

        return np.asarray(
            coordinates,
            dtype=float
        ).reshape(-1, 2)

    def get_spatial_shape(self, level):

        level_obj = self.image[level]

        names = level_obj.zarr.metadata.dimension_names
        mapping = level_obj._dimension_name_mapping

        y_idx = names.index(mapping["Y"])
        x_idx = names.index(mapping["X"])

        height = level_obj.zarr.shape[y_idx]
        width = level_obj.zarr.shape[x_idx]

        return height, width

    def scale_polygons(
        self,
        polygons,
        reference_level,
        target_level
    ):

        ref_h, ref_w = self.get_spatial_shape(reference_level)
        target_h, target_w = self.get_spatial_shape(target_level)

        scale_x = (target_w - 1) / (ref_w - 1)
        scale_y = (target_h - 1) / (ref_h - 1)

        scaled_polygons = []

        for polygon in polygons:

            polygon = polygon.copy()

            polygon[:, 0] *= scale_x
            polygon[:, 1] *= scale_y

            scaled_polygons.append(polygon)

        return scaled_polygons

    def get_polygons(self, target_level=0, timeout=None):

        paths = self.roi_results.get(timeout=timeout)

        polygons = [
            self.parse_polygon(path)
            for path in paths
        ]

        polygons = self.scale_polygons(
            polygons,
            self.reference_level,
            target_level
        )

        return [
            xr.DataArray(
                polygon,
                dims=("vertex", "coordinate"),
                coords={
                    "coordinate": ["X", "Y"],
                },
                name=f"ROI_{i}"
            )
            for i, polygon in enumerate(polygons)
        ]