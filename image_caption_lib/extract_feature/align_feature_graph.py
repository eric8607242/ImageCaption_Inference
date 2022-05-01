import numpy as np

def get_grids_by_corner(corners, grid_size):
    top_left, top_right, bot_left, bot_right = corners
    res = np.zeros((grid_size * grid_size))

    width = top_right - top_left + 1
    for i in range(top_left, bot_left+1, grid_size):
        res[i: i+width] = 1

    return res

def lower_bound(nums, target):
    start = 0
    end = len(nums) - 1
    pos = 0
    
    while start <= end:
        mid = int((start+end)/2)
        if nums[mid] <= target:
            pos = mid
            start = mid + 1
        else:
            end = mid - 1
    return pos

def find_corner(box, image_size, grid_size):
    grids = np.arange(grid_size) / grid_size

    h, w = image_size
    x_min, y_min, x_max, y_max = box
    x_min, y_min, x_max, y_max = x_min/w, y_min/h, x_max/w, y_max/h

    res = []
    # top left
    x1 = lower_bound(grids, x_min)
    y1 = lower_bound(grids, y_min)
    res.append(y1 * grid_size+x1)

    # top right
    x2 = lower_bound(grids, x_max)
    res.append(y1 * grid_size+x2)

    # bot left
    y3 = lower_bound(grids, y_max)
    res.append(y3 * grid_size+x1)
        
    # bot right
    res.append(y3 * grid_size+y3)
    return res

def extract_align_graph(region_features, grid_features, grid_size=7):
    mask = np.zeros((len(region_features["boxes"]), grid_size*grid_size))

    for i, box in enumerate(region_features["boxes"]):
        res = find_corner(box, region_features["size"][0], grid_size)
        grids = get_grids_by_corner(res, grid_size)

        mask[i] = grids

    align_graphs = {
        "mask": mask
    }
    return align_graphs
