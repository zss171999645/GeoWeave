import numpy as np

LABEL_IMGSEG_NAME_MAPPING = {
    0: 'road',
    1 : 'sidewalk',
    2 : 'vegetation',
    3 : 'terrain',
    4 : 'pole',
    5 : 'traffic_sign',
    6 : 'traffic_light',
    7 : 'Sign_Line',
    8 : 'lane_marking',
    9 : 'person',
    10 : 'rider',
    11 : 'bicycle',
    12 : 'motorcycle',
    13 : 'tricycle',
    14 : 'car',
    15 : 'truck',
    16 : 'bus',
    17 : 'train',
    18 : 'building',
    19 : 'fence',
    20 : 'sky',
    21 : 'Traffic_Cone',
    22 : 'Bollard',
    23 : 'Guide_Post',
    24 : 'Crosswalk_Line',
    25 : 'Traffic_Arrow',
    26 : 'Guide_Line',
    27 : 'Stop_Line',
    28 : 'Slow_Down_Triangle',
    29 : 'Speed_Sign',
    30 : 'Diamond',
    31 : 'BicycleSign',
    32 : 'SpeedBumps',
    33 : 'no_forward_marker',
    34 : 'parking_rod',
    35 : 'parking_lock',
    36 : 'traversable_obstruction',
    37: 'untraversable_obstruction',
    38 : 'mask',
    39 : 'other',
}

INV_LABEL_IMGSEG_NAME_MAPPING = {v: k for k, v in LABEL_IMGSEG_NAME_MAPPING.items()}
imgseg_inv_func = np.vectorize(INV_LABEL_IMGSEG_NAME_MAPPING.__getitem__)

dynamic_element_name = [
    'person',
    'rider',
    'bicycle',
    'motorcycle',
    'tricycle',
    'car',
    'truck',
    'bus',
    'train',
]

DYNAMIC_ELEMENT_ID = imgseg_inv_func(dynamic_element_name)