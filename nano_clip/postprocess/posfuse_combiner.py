import numpy as np
from typing import List, Dict, Optional
from utils.logger import default_logger as logger

class AdvancedPosFuseCombinerV2:
    """
    Advanced PosFuse with multiple combination strategies for new JSON format
    """

    def __init__(self, strategy: str = "weighted_sum", decay_factor: float = 0.8):
        """
        Initialize with different combination strategies
        
        Args:
            strategy: "weighted_sum", "max_fusion", "min_fusion", "rank_fusion", "confidence_weighted"
            decay_factor: Position probability decay factor
        """
        self.strategy = strategy
        self.decay_factor = decay_factor

    def create_position_probabilities(self, num_positions: int, strategy: str = "exponential") -> np.ndarray:
        """
        Create position probabilities with different decay strategies
        """
        positions = np.arange(1, num_positions + 1)

        if strategy == "exponential":
            probs = np.power(self.decay_factor, positions - 1)
        elif strategy == "linear":
            probs = (num_positions - positions + 1) / num_positions
        elif strategy == "logarithmic":
            probs = 1.0 / np.log(positions + 1)
        elif strategy == "reciprocal":
            probs = 1.0 / positions
        else:
            probs = np.power(self.decay_factor, positions - 1)

        return probs / np.sum(probs)

    def score_ranking_from_json(self, image_data: List[Dict], prob_strategy: str = "exponential") -> Dict[str, float]:
        """
        Score ranking from JSON format with image_name and overall_score
        
        Args:
            image_data: List of dicts with 'image_name' and 'overall_score'
            prob_strategy: Strategy for position probabilities
        """
        if not image_data:
            return {}

        image_names = [item['image_name'] for item in image_data]
        overall_scores = [item['overall_score'] for item in image_data]

        position_probs = self.create_position_probabilities(len(image_names), prob_strategy)

        # Normalize overall scores
        max_score = max(overall_scores) if overall_scores else 1.0
        min_score = min(overall_scores) if overall_scores else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0
        normalized_scores = [(score - min_score) / score_range for score in overall_scores]

        scores = {}
        for i, image_name in enumerate(image_names):
            position_score = position_probs[i]
            confidence_weight = normalized_scores[i]

            if self.strategy == "confidence_weighted":
                combined_score = position_score * (0.3 + 0.7 * confidence_weight)
            else:
                combined_score = 0.6 * position_score + 0.4 * confidence_weight

            scores[image_name] = combined_score

        return scores

    def score_ranking_from_csv(self, ranked_list: List[str], prob_strategy: str = "exponential") -> Dict[str, float]:
        """
        Tính điểm ranking từ danh sách image trong CSV (chỉ tên ảnh).
        """
        if not ranked_list:
            return {}

        position_probs = self.create_position_probabilities(len(ranked_list), prob_strategy)
        scores = {image_name: position_probs[i] for i, image_name in enumerate(ranked_list)}

        return scores

    def combine_multiple_scores(self, score_dicts: List[Dict[str, float]], 
                                weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Combine scores using different fusion strategies
        """
        if not score_dicts:
            return {}

        if weights is None:
            weights = [1.0] * len(score_dicts)

        all_docs = set()
        for score_dict in score_dicts:
            all_docs.update(score_dict.keys())

        combined_scores = {}

        if self.strategy == "weighted_sum" or self.strategy == "confidence_weighted":
            for doc_id in all_docs:
                total_score = 0.0
                total_weight = 0.0
                for score_dict, weight in zip(score_dicts, weights):
                    if doc_id in score_dict:
                        total_score += weight * score_dict[doc_id]
                        total_weight += weight
                combined_scores[doc_id] = total_score / total_weight if total_weight > 0 else 0.0

        elif self.strategy == "max_fusion":
            for doc_id in all_docs:
                combined_scores[doc_id] = max(score_dict.get(doc_id, 0.0) for score_dict in score_dicts)

        elif self.strategy == "min_fusion":
            for doc_id in all_docs:
                scores_found = [score_dict.get(doc_id) for score_dict in score_dicts if doc_id in score_dict]
                combined_scores[doc_id] = min(scores_found) if len(scores_found) == len(score_dicts) else 0.0

        elif self.strategy == "rank_fusion":
            k = 60
            for doc_id in all_docs:
                rrf_score = 0.0
                for score_dict in score_dicts:
                    if doc_id in score_dict:
                        sorted_docs = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
                        rank = next((i + 1 for i, (doc, _) in enumerate(sorted_docs) if doc == doc_id), len(sorted_docs) + 1)
                        rrf_score += 1.0 / (k + rank)
                combined_scores[doc_id] = rrf_score

        else:  # Mặc định weighted_sum
            for doc_id in all_docs:
                total_score = 0.0
                total_weight = 0.0
                for score_dict, weight in zip(score_dicts, weights):
                    if doc_id in score_dict:
                        total_score += weight * score_dict[doc_id]
                        total_weight += weight
                combined_scores[doc_id] = total_score / total_weight if total_weight > 0 else 0.0

        return combined_scores
