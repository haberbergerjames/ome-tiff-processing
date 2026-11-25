import numpy as np
import plotly.express as px

from dash import Dash, dcc, html, Input, Output, State, no_update


class ImageSelection:

    def __init__(self, polygons, image_shape):
        self.polygons = polygons
        self.image_shape = image_shape

    def __len__(self):
        return len(self.polygons)


class PolygonSelector:

    def __init__(self, image):

        self.image = np.asarray(image)
        self.height, self.width = self.image.shape[:2]

        self.selection = None

        self.app = Dash(__name__)
        self._build_figure()
        self._build_layout()
        self._register_callbacks()

    def _build_figure(self):

        self.figure = px.imshow(
            self.image,
            binary_string=True,
        )

        self.figure.update_layout(
            dragmode="drawclosedpath",
            newshape=dict(
                line=dict(color="lime", width=3),
                fillcolor="rgba(0,255,0,0.15)",
            ),
        )

    def _build_layout(self):

        self.app.layout = html.Div([

            dcc.Graph(
                id="image",
                figure=self.figure,
                config={
                    "modeBarButtonsToAdd": [
                        "drawclosedpath",
                        "eraseshape",
                    ]
                },
            ),

            dcc.Store(id="polygon-data"),

            html.Div([
                html.Button("Draw", id="draw"),
                html.Button("Navigate", id="navigate"),
                html.Button("Reset view", id="reset"),
                html.Button("Done", id="done"),
            ]),

            html.Div(id="status"),

        ])

    def _register_callbacks(self):

        @self.app.callback(
            Output("image", "figure"),
            Input("draw", "n_clicks"),
            Input("navigate", "n_clicks"),
            Input("reset", "n_clicks"),
            State("image", "figure"),
            prevent_initial_call=True,
        )
        def update_figure(draw, navigate, reset, figure):

            from dash import ctx

            if ctx.triggered_id == "draw":
                figure["layout"]["dragmode"] = "drawclosedpath"

            elif ctx.triggered_id == "navigate":
                figure["layout"]["dragmode"] = "pan"

            elif ctx.triggered_id == "reset":
                figure["layout"]["xaxis"]["autorange"] = True
                figure["layout"]["yaxis"]["autorange"] = True

            return figure

        @self.app.callback(
            Output("status", "children"),
            Input("image", "relayoutData"),
            Input("done", "n_clicks"),
            prevent_initial_call=True,
        )
        def update_selection(relayout_data, done):

            from dash import ctx

            if ctx.triggered_id == "image":
                return "Selection updated"

            if ctx.triggered_id == "done":
                if not hasattr(self, "_latest_shapes"):
                    return "No polygons selected"

                polygons = self._parse_polygons(
                    self._latest_shapes
                )

                self.selection = ImageSelection(
                    polygons=polygons,
                    image_shape=self.image.shape,
                )

                return f"Selected {len(polygons)} regions"

        # Track the complete shape state separately.
        @self.app.callback(
            Output("polygon-data", "data"),
            Input("image", "relayoutData"),
            State("polygon-data", "data"),
            prevent_initial_call=True,
        )
        def capture_shapes(data, previous):

            if not data:
                return no_update

            if "shapes" in data:
                self._latest_shapes = data["shapes"]
                return data["shapes"]

            # Incremental shape edits are handled in a later pass.
            return no_update

    def _parse_polygons(self, shapes):

        import re

        polygons = []

        for shape in shapes:

            if shape.get("type") != "path":
                continue

            numbers = re.findall(
                r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?",
                shape["path"],
            )

            coords = np.asarray(
                numbers, dtype=float
            ).reshape(-1, 2)

            polygons.append(coords)

        return polygons

    def show(self):

        self.app.run(jupyter_mode="inline", debug=False)
