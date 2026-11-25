import numpy as np
import plotly.express as px

from dash import Dash, dcc, html, Input, Output, State, no_update


class ImageSelection:

    def __init__(self, polygons, image_shape):
        self.polygons = polygons
        self.image_shape = image_shape

    def __len__(self):
        return len(self.polygons)