import os
import argparse
import numpy as np
import json


parser = argparse.ArgumentParser(description='TSClusteringX')

parser.add_argument('--dataset', type=str, default='ucr_uea', help='dataset name')
parser.add_argument('--start', type=int, default=1, help='start number of dataset')
parser.add_argument('--end', type=int, default=128, help='end number of dataset')
parser.add_argument('--dataset_path', type=str, default='../data/UCR2018/', help='path to the dataset')
parser.add_argument('--model', type=str, default='agglomerative', help='name of the model')
parser.add_argument('--distance', type=str, default=None, help='distance measure')
parser.add_argument('--parameter_settings', type=str, default=None, help='parameter settings')
parser.add_argument('--metrics', type=str, nargs='+', default=None, help='list of metrics')
parser.add_argument('--experiment', type=str, help='experiment name')

args = parser.parse_args()


if args.dataset == 'ucr_uea':
    if args.start is None or args.end is None or args.dataset_path is None:
        parser.error("For 'ucr_uea' dataset, --start, and --end are required.")
else:
    args.start = None
    args.end = None


from dataloaders.dataloader import DataLoaderFactory

dataloader = DataLoaderFactory.get_dataloader(args.dataset, args.dataset_path)

results_dir = f"results/{args.model}"
os.makedirs(results_dir, exist_ok=True)

all_results = []
for i, sub_dataset_name in enumerate(sorted(os.listdir(args.dataset_path), key=str.lower)[args.start-1:args.end]):
    print(sub_dataset_name)
    ts, labels, nclusters = dataloader.load(sub_dataset_name)

    distance_matrix = None
    if args.distance:
        from distances.distance import DistanceFactory
        distance_measure = DistanceFactory.get_distance(args.distance)
        if distance_measure is not None:
            distance_matrix = distance_measure.compute(ts)
            print(f"Distance matrix shape: {distance_matrix.shape}")
        else:
            print(f"Warning: Unknown distance '{args.distance}', proceeding without distance matrix")

    from models.model import ModelFactory

    # Load model parameters if provided
    model_params = {}
    if args.parameter_settings:
        if os.path.exists(args.parameter_settings):
            try:
                with open(args.parameter_settings, 'r', encoding='utf-8') as f:
                    model_params = json.load(f)
                print(f"Loaded parameters from {args.parameter_settings}: {model_params}")
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in parameter settings file {args.parameter_settings}: {e}")
                print("Using default parameters")
            except Exception as e:
                print(f"Warning: Error reading parameter settings file {args.parameter_settings}: {e}")
                print("Using default parameters")
        else:
            print(f"Warning: Parameter settings file not found: {args.parameter_settings}")
            print("Using default parameters")

    model = ModelFactory.get_model(args.model, nclusters, model_params, distance_name=args.distance, distance_matrix=distance_matrix)
    predicted_labels, elapsed = model.fit_predict(ts)

    print(predicted_labels)
    print(elapsed)
    print(args.metrics)

    from metrics.metric import ClusterMetrics
    metrics = ClusterMetrics(labels, predicted_labels)

    # Calculate metrics
    ri_score = metrics.rand_score()
    ari_score = metrics.adjusted_rand_score()
    nmi_score = metrics.normalized_mutual_information()
    
    print('ri', ri_score)
    print('ari', ari_score)
    print('nmi', nmi_score)
    
    # Prepare result data for this dataset
    result_data = {
        'dataset_name': sub_dataset_name,
        'predicted_labels': predicted_labels,
        'original_labels': labels,
        'elapsed_time': elapsed,
        'distance_measure': args.distance,
        'metrics': {}
    }
    
    # Add requested metrics to results
    if args.metrics:
        metric_functions = {
            'RI': ri_score,
            'ARI': ari_score, 
            'NMI': nmi_score
        }
        
        for metric in args.metrics:
            if metric.upper() in metric_functions:
                result_data['metrics'][metric.upper()] = metric_functions[metric.upper()]
    
    # Store result for this dataset
    all_results.append(result_data)

# Save all results to file
experiment_file = os.path.join(results_dir, f"experiment-{args.experiment}.npy")
np.save(experiment_file, all_results)
print(f"Results saved to {experiment_file}")
