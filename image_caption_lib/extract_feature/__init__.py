from .align_feature_graph import extract_align_graph
from .extract_grid_feature import extract_grid_feature
from .extract_region_feature import extract_region_feature
from .utils import inititalize_extrace_model, wrap_image_to_inputs

def extract_features(model, path_to_image):
    inputs = wrap_image_to_inputs(model, path_to_image)

    grid_features = extract_grid_feature(model, inputs)
    region_features = extract_region_feature(model, inputs)
    align_graphs = extract_align_graph(region_features, grid_features)

    return {
        "grid_features": grid_features,
        "region_features": region_features,
        "align_graphs": align_graphs
    }
