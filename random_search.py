# Import
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import itertools
import random
import pandas as pd
import csv
from time import time
import threading


# Ground truth
ground_truth = {
    # gt_from_0_to_9
    "scene_0":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_1":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # mega dubbio sul model 18, se è 1 o 2
    "scene_2":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "scene_3":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    "scene_4":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
    "scene_5":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_6":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "scene_7":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    "scene_8":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_9":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],  # dubbio sul model 19, se è 2 o 4

    # gt_from_10_to_19
    "scene_10": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    "scene_11": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_12": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_13": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_14": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_15": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # same as 16 and 17 but with different light or color scale
    "scene_16": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # same as 15 and 17 but with different light or color scale
    "scene_17": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # same as 15 and 16 but with different light or color scale
    "scene_18": [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NOTE: model_11 instances are 2 (one ruined, considered 1); model_12 instances are 3 (differences, considered 1)
    "scene_19": [0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # gt
    "scene_20": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_21": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_22": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_23": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_24": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_25": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_26": [2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "scene_27": [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # supersus
    "scene_28": [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

# Models and Scenes
models_path = './dataset/models/'
scenes_path = './dataset/scenes/'
imgs_model = [os.path.join(models_path, f) for f in os.listdir(models_path)]
imgs_scene = [os.path.join(scenes_path, f) for f in os.listdir(scenes_path)]

models = [f'model_{i}.png' for i in range(0, 21)]
scenes = [f'scene_{i}.jpg' for i in range(0, 28)]

imgs_model = [os.path.join(models_path, f) for f in models]
imgs_scene = [os.path.join(scenes_path, f) for f in scenes]

# Scenes categorization
easy_scenes_keys = ["scene_0", "scene_8", "scene_11", "scene_12", "scene_13", "scene_14", "scene_20", "scene_21", "scene_22", "scene_24", "scene_25"]
impossible_scene_keys = ["scene_9", "scene_10", "scene_15", "scene_16", "scene_17", "scene_18", "scene_27"]

hard_scenes_keys = [k for k in ground_truth.keys() if k not in easy_scenes_keys and k not in impossible_scene_keys]

easy_scene_paths = [os.path.join(scenes_path, f"{k}.jpg") for k in easy_scenes_keys]
hard_scene_paths = [os.path.join(scenes_path, f"{k}.jpg") for k in hard_scenes_keys]
impossible_scene_paths = [os.path.join(scenes_path, f"{k}.jpg") for k in impossible_scene_keys]

# Utils 
def is_convex(pts):
    hull = cv2.convexHull(pts)
    return len(hull) == len(pts)

def is_inside_image(pt, img):
    h, w = img.shape[:2]
    if pt[0] < 0 or pt[0] >= w or pt[1] < 0 or pt[1] >= h:
        return False
    return True

def is_inside_polygon(pt, polygon:np.ndarray):
    result = cv2.pointPolygonTest(polygon, pt, False)
    return result >= 0

def rescale_polygon(pts, scale):
    center = np.mean(pts, axis=0)
    for i in range(len(pts)):
        pts[i] = pts[i] + (scale - 1) * (pts[i] - center)
    return pts

# Get key points from models
def get_keypoints(img_model_paths:list[str],
                  hyp_params:dict={}):
    
    # Hyper-parameters:
    nfeatures = hyp_params.get('sift_nfeatures', 0)
    nOctaveLayers = hyp_params.get('sift_nOctaveLayers', 3)
    contrastThreshold = hyp_params.get('sift_contrastThreshold', 0.04)
    edgeThreshold = hyp_params.get('sift_edgeThreshold', 10)
    sigma = hyp_params.get('sift_sigma', 0.5)

    # SIFT detector
    sift = cv2.SIFT_create(nfeatures=nfeatures,
                           nOctaveLayers=nOctaveLayers,
                           contrastThreshold=contrastThreshold,
                           edgeThreshold=edgeThreshold,
                           sigma=sigma)
    
    # keypoints dict
    kp_dict = {}
    
    for img_model_path in img_model_paths:

        image_key = f"model_{img_model_path.split('_')[1].split('.')[0]}"
        
        img_model = cv2.imread(img_model_path)
        
        kp_model = sift.detect(img_model)
        kp_model, des_model = sift.compute(img_model, kp_model)

        kp_dict[image_key] = (kp_model, des_model)

    return kp_dict

# Locate a book in a scene
def locate_book_from_scene_kp_desc(img_model,
                                   kp_model,
                                   des_model,
                                   img_scene,
                                   kp_scene,
                                   des_scene,
                                   matcher,
                                   hyp_params:dict={}):

    instance = None

    good_matches_threshold = hyp_params.get('good_matches_threshold', 0.75)
    min_match_count = hyp_params.get('min_match_count', 50)

    # Finding matches using KNN
    matches = matcher.knnMatch(des_model, des_scene, k=2)

    # Filtering false matches using the Lowe's ratio test
    good = [m for m, n in matches if m.distance < good_matches_threshold * n.distance]

    # Corners of the model image
    h, w = img_model.shape[:2]
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1, 1, 2)

    if len(good) > min_match_count: # > min_match_percentage * len(kp_model)
        
        # Building the correspondence arrays of good matches
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Using RANSAC to estimate a robust homography.
        # It returns the homography M and a mask for the discarded points.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Mask of discarded point used in visualization
        matches_mask = mask.ravel().tolist()

        # Corners of the model image
        h, w = img_model.shape[:2]
        h_t, w_t = img_scene.shape[:2]
        pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1, 1, 2)

        # Projecting the corners into the scene image
        dst = cv2.perspectiveTransform(pts, M)

        # Saving position and area info
        instance = {
            "dst" : dst,
            "top_left" : (int(dst[0][0][0]), int(dst[0][0][1])),
            "bottom_left" : (int(dst[1][0][0]), int(dst[1][0][1])),
            "bottom_right" : (int(dst[2][0][0]), int(dst[2][0][1])),
            "top_right" : (int(dst[3][0][0]), int(dst[3][0][1])),
            "area": int(cv2.contourArea(dst))
        }

    return instance

# Locate all books in a scene
def locate_books_in_scene_JJJJ(img_scene_path:str,
                          kp_des_models: dict,
                          hyp_params:dict={}):
    
    # Hyper-parameters:
    nfeatures = hyp_params.get('sift_nfeatures', 0)
    nOctaveLayers = hyp_params.get('sift_nOctaveLayers', 3)
    contrastThreshold = hyp_params.get('sift_contrastThreshold', 0.04)
    edgeThreshold = hyp_params.get('sift_edgeThreshold', 10)
    sigma = hyp_params.get('sift_sigma', 0.5)

    mask_scale = hyp_params.get('mask_scale', 1.0)

    # SIFT detector
    sift = cv2.SIFT_create(nfeatures=nfeatures,
                           nOctaveLayers=nOctaveLayers,
                           contrastThreshold=contrastThreshold,
                           edgeThreshold=edgeThreshold,
                           sigma=sigma)

    # Read images
    img_scene = cv2.imread(img_scene_path)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Keypoints and descriptors
    kp_scene = sift.detect(img_scene)
    kp_scene, des_scene = sift.compute(img_scene, kp_scene)

    books_instances = {}
    
    for model_key, (kp_model, des_model) in kp_des_models.items():
        
        instances = []

        img_model = cv2.imread(os.path.join(models_path, model_key+'.png'))

        while True:

            instance = locate_book_from_scene_kp_desc(img_model=img_model,
                                                      kp_model=kp_model,
                                                      des_model=des_model,
                                                      img_scene=img_scene,
                                                      kp_scene=kp_scene,
                                                      des_scene=des_scene,
                                                      matcher=flann,
                                                      hyp_params=hyp_params)
            if instance is not None:

                instances.append(instance)

                dst = instance["dst"]

                # We create a mask (scaled) to avoid detecting the same object again
                dst_scaled = rescale_polygon(dst, mask_scale)
                mask_polygon = np.int32(dst_scaled)

                # Remove from the keypoints the ones that are inside the scaled polygon and the relative descriptors
                new_kp_scene = []
                new_des_scene = []
                for i in range(len(kp_scene)):
                    if not is_inside_image(kp_scene[i].pt, img_scene):
                        continue
                    if is_inside_polygon(kp_scene[i].pt, dst_scaled):
                        continue
                    new_kp_scene.append(kp_scene[i])
                    new_des_scene.append(des_scene[i])
                
                kp_scene = new_kp_scene
                des_scene = np.array(new_des_scene)

            else:
                
                break
        
        books_instances[model_key] = instances

    return books_instances

# Param grid
param_grid = {
    'sift_nfeatures': [0],
    'sift_nOctaveLayers': list(range(3, 5)),
    'sift_contrastThreshold': list(np.linspace(0.02, 0.06, 100)),
    'sift_edgeThreshold': list(np.linspace(5, 15, 100)),
    'sift_sigma': list(np.linspace(0.4, 0.8, 100)),
    'min_match_count': list(range(40, 60)),
    'good_matches_threshold': list(np.linspace(0.6, 0.9, 100)),
    'mask_scale': [1]   
}

# Evaluate function
def evaluate(scenes,
             models,
             params:dict={},
             MAX_ERRORS = 25):

    result = {"value": None}

    def target():
        try:
            easy_scenes, hard_scenes = scenes

            # Get models kp, des
            kp_des_models = get_keypoints(models, params)

            errors = 0

            # Scenes loop
            for i, s in enumerate(easy_scenes + hard_scenes):

                # Extract scene index
                scene_index = s.split('_')[1].split('.')[0]

                # Read scene image
                img_scene = cv2.imread(s)
                img_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2RGB)

                # Get ground truth counts for the scene
                gt_counts = ground_truth[f'scene_{scene_index}']

                # Locate books in the scene
                books = locate_books_in_scene_JJJJ(s, kp_des_models, params)

                # Loop on models found in the scene
                for m_idx, (model_key, instances) in enumerate(books.items()):

                    # Extract book index
                    book_index = model_key.split('_')[1]

                    # Get expected count from ground truth
                    expected_count = gt_counts[int(book_index)]

                    if expected_count != len(instances) and i < len(easy_scenes):
                        print(f'\tError in a easy scene :(', flush=True)
                        result["value"] = 2 * MAX_ERRORS
                        return

                    errors += abs(expected_count - len(instances))

                    if errors >= MAX_ERRORS:
                        print(f'\tToo many errors womp womp', flush=True)
                        result["value"] = MAX_ERRORS
                        return

            print(f'\tTotal errors: {errors}')
            result["value"] = errors
        except Exception as e:
            print("\tSomething went wrong", flush=True)
            result["value"] = 4 * MAX_ERRORS

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=100)
    if thread.is_alive():
        print("\tTimeout", flush=True)
        return 3 * MAX_ERRORS
    return result["value"]

# Random search
def random_search(param_grid, out_file, max_evals):
    """Random search for hyperparameter optimization"""

    results = []

    try:
        for i in range(max_evals):
            # Choose random hyperparameters
            hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            
            print(f'Random eval {i+1}: {hyperparameters}', flush=True)

            # Evaluate randomly selected hyperparameters
            eval_results = evaluate((easy_scene_paths, hard_scene_paths + impossible_scene_paths), imgs_model, hyperparameters)

            results.append({'score': eval_results, 'params': hyperparameters, 'iteration': i})
    except KeyboardInterrupt:
        print("Random search interrupted", flush=True)

    # Create DataFrame from results
    df_results = pd.DataFrame(results)
    # Sort with best score on top
    df_results.sort_values('score', ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)
    return df_results

def main():
    MAX_EVALS = 300

    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)

    out_file = f'random_search_{time()}.csv'

    out_path = os.path.join(out_dir, out_file)

    random_results = random_search(param_grid, out_path, MAX_EVALS)

    # save random_results to csv
    random_results.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()

