import csv
import ast
import re
import sys

def parse_params(params_str):
    # Replace np.float64(x) with x
    params_str = re.sub(r'np\.float64\(([^)]+)\)', r'\1', params_str)
    # Convert string dict to actual dict
    params = ast.literal_eval(params_str)
    return params

def convert_csv(input_path, output_path):
    with open(input_path, newline='') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = [
            'score', 'iteration', 'sift_nfeatures', 'sift_nOctaveLayers',
            'sift_contrastThreshold', 'sift_edgeThreshold', 'sift_sigma',
            'min_match_count', 'good_matches_threshold', 'mask_scale'
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            params = parse_params(row['params'])
            out_row = {
                'score': row['score'],
                'iteration': row['iteration'],
                'sift_nfeatures': params.get('sift_nfeatures'),
                'sift_nOctaveLayers': params.get('sift_nOctaveLayers'),
                'sift_contrastThreshold': params.get('sift_contrastThreshold'),
                'sift_edgeThreshold': params.get('sift_edgeThreshold'),
                'sift_sigma': params.get('sift_sigma'),
                'min_match_count': params.get('min_match_count'),
                'good_matches_threshold': params.get('good_matches_threshold'),
                'mask_scale': params.get('mask_scale'),
            }
            writer.writerow(out_row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python old_results_converter.py input.csv output.csv")
        sys.exit(1)
    convert_csv(sys.argv[1], sys.argv[2])