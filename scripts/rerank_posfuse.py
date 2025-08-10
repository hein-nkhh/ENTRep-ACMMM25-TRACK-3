import os
from nano_clip.postprocess.utils import test_multiple_strategies_v2, compare_strategies_v2, create_submission_json
from utils.config import load_config

cfg = load_config()

def main():
    print("=" * 60)
    print("ADVANCED POSFUSE MULTI-STRATEGY TESTING V2")
    print("=" * 60)
    
    csv_file = os.path.join(cfg['data']['RESULT_TOP_K'], 'image_retrieval_results.csv')
    json_file = r"E:\Project\ENTRep-ACMMM25-TRACK-3\data\Postpre\combined_reranked_results_avg.json"

    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return

    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        return

    print("Input files found:")
    print(f"CSV: {csv_file}")
    print(f"JSON: {json_file}")
    print()

    strategy_results = test_multiple_strategies_v2(csv_file, json_file)
    if not strategy_results:
        print("No results from strategy testing.")
        return
    
    os.makedirs(cfg['data']['RERANK_POSFUSE_DIR'], exist_ok=True)
    best_strategy = compare_strategies_v2(strategy_results)
    if best_strategy:
        create_submission_json(strategy_results, best_strategy, output_file=cfg['data']['RERANK_POSFUSE_JSON_DIR'])
    else:
        print("Failed to determine best strategy.")

if __name__ == "__main__":
    main()
