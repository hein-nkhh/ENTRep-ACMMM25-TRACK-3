import pandas as pd
import json
import ast
from typing import Dict, List
from .posfuse_combiner import AdvancedPosFuseCombinerV2

def test_multiple_strategies_v2(csv_file: str, json_file: str) -> Dict[str, pd.DataFrame]:
    print("Loading data for multi-strategy testing...")

    # Load CSV
    try:
        df = pd.read_csv(csv_file)
        csv_results = {}
        for _, row in df.iterrows():
            try:
                csv_results[row['text']] = ast.literal_eval(row['image_name'])
            except Exception as e:
                print(f"Error parsing CSV row: {e}")
                continue
        print(f"Loaded {len(csv_results)} queries from CSV")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return {}

    # Load JSON
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            json_results = json.load(f)
        print(f"Loaded {len(json_results)} queries from JSON")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}

    strategies = ["weighted_sum", "max_fusion", "min_fusion", "rank_fusion", "confidence_weighted"]
    all_results = {}

    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        combiner = AdvancedPosFuseCombinerV2(strategy=strategy, decay_factor=0.8)
        strategy_results = []
        processed_count = 0
        error_count = 0

        for query_text, csv_images in csv_results.items():
            try:
                if query_text not in json_results:
                    continue

                json_image_data = json_results[query_text]
                if not json_image_data:
                    continue

                csv_scores = combiner.score_ranking_from_csv(csv_images)
                json_scores = combiner.score_ranking_from_json(json_image_data)

                combined_scores = combiner.combine_multiple_scores([csv_scores, json_scores], weights=[0.4, 0.6])

                if combined_scores:
                    final_ranking = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                    final_images = [img for img, _ in final_ranking]

                    csv_top5 = set(csv_images[:5])
                    json_top5 = set([item['image_name'] for item in json_image_data][:5])
                    final_top5 = set(final_images[:5])

                    strategy_results.append({
                        'query': query_text,
                        'strategy': strategy,
                        'final_ranking': final_images,
                        'csv_overlap': len(csv_top5 & final_top5),
                        'json_overlap': len(json_top5 & final_top5),
                        'total_images': len(final_images),
                        'csv_original': csv_images,
                        'json_original': [item['image_name'] for item in json_image_data],
                        'json_scores': [item['overall_score'] for item in json_image_data]
                    })

                    processed_count += 1

            except Exception as e:
                error_count += 1
                print(f"Error processing query '{query_text[:50]}...': {e}")

        all_results[strategy] = pd.DataFrame(strategy_results)
        print(f"Processed {processed_count} queries successfully, {error_count} errors")

    return all_results


def compare_strategies_v2(strategy_results: Dict[str, pd.DataFrame]) -> str:
    print("\n" + "="*80)
    print("STRATEGY COMPARISON ANALYSIS (V2)")
    print("="*80)

    if not strategy_results:
        print("No strategy results to compare!")
        return None

    summary_stats = {}
    for strategy, df in strategy_results.items():
        if not df.empty:
            summary_stats[strategy] = {
                'avg_csv_overlap': df['csv_overlap'].mean(),
                'avg_json_overlap': df['json_overlap'].mean(),
                'num_queries': len(df),
                'std_csv_overlap': df['csv_overlap'].std(),
                'std_json_overlap': df['json_overlap'].std(),
                'total_score': df['csv_overlap'].mean() + df['json_overlap'].mean()
            }

    if not summary_stats:
        print("No valid strategy results found!")
        return None

    print("\nStrategy Performance Summary:")
    print("-" * 90)
    print(f"{'Strategy':<20} {'CSV Overlap':<15} {'JSON Overlap':<16} {'Total Score':<12} {'Std Dev':<15} {'Queries':<8}")
    print("-" * 90)

    for strategy, stats in summary_stats.items():
        std_combined = (stats['std_csv_overlap']**2 + stats['std_json_overlap']**2)**0.5
        print(f"{strategy:<20} {stats['avg_csv_overlap']:<15.2f} {stats['avg_json_overlap']:<16.2f} {stats['total_score']:<12.2f} {std_combined:<15.2f} {stats['num_queries']:<8}")

    best_strategy = max(summary_stats.keys(), key=lambda x: summary_stats[x]['total_score'])
    print(f"\nBest performing strategy: {best_strategy}")
    print(f"Combined overlap score: {summary_stats[best_strategy]['total_score']:.2f}")

    return best_strategy


def create_submission_json(strategy_results: Dict[str, pd.DataFrame], 
                        best_strategy: str,
                        output_file: str = "combine_gemini.json") -> None:
    if not strategy_results or not best_strategy or best_strategy not in strategy_results:
        print("Invalid inputs for creating submission.")
        return None

    print(f"\nCreating submission from best strategy: {best_strategy}")
    submission = {}
    best_df = strategy_results[best_strategy]

    for _, row in best_df.iterrows():
        query = row['query']
        final_ranking = row['final_ranking']
        if final_ranking:
            submission[query] = final_ranking[0]

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(submission, f, indent=4, ensure_ascii=False)
        print(f"Submission saved to: {output_file}")
        print(f"Total queries in submission: {len(submission)}")
        return submission
    except Exception as e:
        print(f"Error saving submission: {e}")
        return None
