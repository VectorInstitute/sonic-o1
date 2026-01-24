"""
T1 Metrics: Video Summarization Evaluation
Computes ROUGE-L, CIDEr, text similarity, and LLM-as-Judge scores.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


class T1Metrics:
    """Compute metrics for T1: Video Summarization."""

    def __init__(
        self,
        use_llm_judge: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        judge_name: str = "gpt",
    ):
        """
        Initialize T1 metrics.

        Args:
            use_llm_judge: Whether to use LLM judge evaluation
            embedding_model: Model for computing text similarity
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.cider_scorer = Cider()
        self.embedding_model = SentenceTransformer(embedding_model)

        self.use_llm_judge = use_llm_judge
        logger.info(f"LLM Judge name: {judge_name}")
        if use_llm_judge:
            if judge_name == "gpt":
                from llm_judge_gpt import LLMJudge

                self.llm_judge = LLMJudge()
            elif judge_name == "qwen":
                from llm_judge_qwen import LLMJudge

                self.llm_judge = LLMJudge()
            else:
                raise ValueError(f"Unknown judge name: {judge_name}")

    def compute_rouge_l(self, reference: str, prediction: str) -> float:
        """
        Compute ROUGE-L F1 score.

        Args:
            reference: Ground truth text
            prediction: Predicted text

        Returns
        -------
            ROUGE-L F1 score (0-1)
        """
        scores = self.rouge_scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure

    def compute_cider(self, references: List[str], predictions: List[str]) -> float:
        """
        Compute CIDEr score for a set of summaries.

        Args:
            references: List of ground truth summaries
            predictions: List of predicted summaries

        Returns
        -------
            Average CIDEr score
        """
        # CIDEr expects dict format: {id: [text]}
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}

        score, scores = self.cider_scorer.compute_score(gts, res)
        return float(score)

    def compute_text_similarity(self, reference: str, prediction: str) -> float:
        """
        Compute cosine similarity between embeddings.

        Args:
            reference: Ground truth text
            prediction: Predicted text

        Returns
        -------
            Cosine similarity (0-1)
        """
        ref_embedding = self.embedding_model.encode([reference])
        pred_embedding = self.embedding_model.encode([prediction])

        similarity = cosine_similarity(ref_embedding, pred_embedding)[0][0]
        return float(similarity)

    def evaluate_entry(
        self, ground_truth: Dict[str, Any], prediction: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single video entry.

        Args:
            ground_truth: Ground truth entry
            prediction: Predicted entry

        Returns
        -------
            Dict with all metrics, or None if prediction failed
        """
        # Skip failed predictions
        if "error" in prediction or "outputs" not in prediction:
            logger.warning(
                f"Skipping failed entry: {prediction.get('video_id', 'unknown')}"
            )
            return None

        results = {
            "video_id": ground_truth["video_id"],
            "video_number": ground_truth["video_number"],
        }

        # Extract texts
        gt_detailed = ground_truth.get("summary_detailed", "")
        pred_detailed = prediction["outputs"].get("summary_detailed", "")

        gt_short = " ".join(ground_truth.get("summary_short", []))
        pred_short = " ".join(prediction["outputs"].get("summary_short", []))

        # Compute metrics for detailed summary
        results["detailed"] = {
            "rouge_l": self.compute_rouge_l(gt_detailed, pred_detailed),
            "text_similarity": self.compute_text_similarity(gt_detailed, pred_detailed),
        }

        # LLM Judge for detailed summary
        if self.use_llm_judge and gt_detailed and pred_detailed:
            try:
                llm_eval = self.llm_judge.evaluate(
                    question="Provide a detailed summary of the video content.",
                    correct_answer=gt_detailed,
                    predicted_answer=pred_detailed,
                    task_type="summarization_detailed",
                )
                results["detailed"]["llm_judge_score"] = llm_eval["score"]
                results["detailed"]["llm_judge_justification"] = llm_eval[
                    "justification"
                ]
            except Exception as e:
                logger.error(f"LLM judge failed for {results['video_id']}: {e}")
                results["detailed"]["llm_judge_score"] = None
                results["detailed"]["llm_judge_justification"] = str(e)

        # Compute metrics for short summary
        results["short"] = {
            "rouge_l": self.compute_rouge_l(gt_short, pred_short),
            "text_similarity": self.compute_text_similarity(gt_short, pred_short),
        }

        # LLM Judge for short summary
        if self.use_llm_judge and gt_short and pred_short:
            try:
                llm_eval = self.llm_judge.evaluate(
                    question="Provide a brief bullet-point summary of the video.",
                    correct_answer=gt_short,
                    predicted_answer=pred_short,
                    task_type="summarization_short",
                )
                results["short"]["llm_judge_score"] = llm_eval["score"]
                results["short"]["llm_judge_justification"] = llm_eval["justification"]
            except Exception as e:
                logger.error(
                    f"LLM judge failed for short summary {results['video_id']}: {e}"
                )
                results["short"]["llm_judge_score"] = None
                results["short"]["llm_judge_justification"] = str(e)

        return results

    def evaluate_topic(
        self, ground_truth_path: Path, prediction_path: Path
    ) -> Dict[str, Any]:
        """
        Evaluate all entries for a topic.

        Args:
            ground_truth_path: Path to ground truth JSON
            prediction_path: Path to prediction JSON

        Returns
        -------
            Dict with aggregated metrics
        """
        # Load data
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)

        with open(prediction_path, "r") as f:
            prediction = json.load(f)

        # Match entries by video_id
        gt_entries = {e["video_id"]: e for e in ground_truth["entries"]}
        pred_entries = {e["video_id"]: e for e in prediction["entries"]}

        # Evaluate each entry
        entry_results = []
        for video_id, gt_entry in gt_entries.items():
            if video_id not in pred_entries:
                logger.warning(f"Missing prediction for video {video_id}")
                continue

            pred_entry = pred_entries[video_id]
            result = self.evaluate_entry(gt_entry, pred_entry)

            # Skip None results from failed entries
            if result is not None:
                entry_results.append(result)

        # Aggregate metrics
        aggregated = self._aggregate_results(entry_results)

        return {
            "topic_id": ground_truth["topic_id"],
            "topic_name": ground_truth["topic_name"],
            "num_evaluated": len(entry_results),
            "aggregated_metrics": aggregated,
            "per_entry_results": entry_results,
        }

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all entries."""
        aggregated = {}

        for summary_type in ["detailed", "short"]:
            metrics = {"rouge_l": [], "text_similarity": [], "llm_judge_score": []}

            for result in results:
                if summary_type in result:
                    metrics["rouge_l"].append(result[summary_type]["rouge_l"])
                    metrics["text_similarity"].append(
                        result[summary_type]["text_similarity"]
                    )

                    if result[summary_type].get("llm_judge_score") is not None:
                        metrics["llm_judge_score"].append(
                            result[summary_type]["llm_judge_score"]
                        )

            aggregated[summary_type] = {
                "rouge_l_mean": float(np.mean(metrics["rouge_l"]))
                if metrics["rouge_l"]
                else 0.0,
                "rouge_l_std": float(np.std(metrics["rouge_l"]))
                if metrics["rouge_l"]
                else 0.0,
                "text_similarity_mean": float(np.mean(metrics["text_similarity"]))
                if metrics["text_similarity"]
                else 0.0,
                "text_similarity_std": float(np.std(metrics["text_similarity"]))
                if metrics["text_similarity"]
                else 0.0,
            }

            if metrics["llm_judge_score"]:
                aggregated[summary_type]["llm_judge_score_mean"] = float(
                    np.mean(metrics["llm_judge_score"])
                )
                aggregated[summary_type]["llm_judge_score_std"] = float(
                    np.std(metrics["llm_judge_score"])
                )

        return aggregated

    def compute_cider_for_topic(
        self, ground_truth_path: Path, prediction_path: Path
    ) -> Dict[str, float]:
        """
        Compute CIDEr scores for a topic.

        Args:
            ground_truth_path: Path to ground truth JSON
            prediction_path: Path to prediction JSON

        Returns
        -------
            Dict with CIDEr scores for detailed and short summaries
        """
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)

        with open(prediction_path, "r") as f:
            prediction = json.load(f)

        gt_entries = {e["video_id"]: e for e in ground_truth["entries"]}
        pred_entries = {e["video_id"]: e for e in prediction["entries"]}

        gt_detailed = []
        pred_detailed = []
        gt_short = []
        pred_short = []

        for video_id, gt_entry in gt_entries.items():
            if video_id in pred_entries:
                pred_entry = pred_entries[video_id]

                # Skip failed predictions
                if "error" in pred_entry or "outputs" not in pred_entry:
                    continue

                gt_detailed.append(gt_entry.get("summary_detailed", ""))
                pred_detailed.append(pred_entry["outputs"].get("summary_detailed", ""))

                gt_short.append(" ".join(gt_entry.get("summary_short", [])))
                pred_short.append(
                    " ".join(pred_entry["outputs"].get("summary_short", []))
                )

        return {
            "cider_detailed": self.compute_cider(gt_detailed, pred_detailed)
            if gt_detailed
            else 0.0,
            "cider_short": self.compute_cider(gt_short, pred_short)
            if gt_short
            else 0.0,
        }


def evaluate_t1_topic(
    ground_truth_path: str,
    prediction_path: str,
    output_path: str,
    use_llm_judge: bool = True,
    judge_name: str = "gpt",
):
    """
    Convenience function to evaluate a single topic.

    Args:
        ground_truth_path: Path to ground truth JSON
        prediction_path: Path to prediction JSON
        output_path: Where to save results
        use_llm_judge: Whether to use LLM judge
    """
    logger.info(f"Evaluating T1: {Path(ground_truth_path).stem}")

    metrics = T1Metrics(use_llm_judge=use_llm_judge, judge_name=judge_name)
    results = metrics.evaluate_topic(
        Path(ground_truth_path),
        Path(prediction_path),
    )

    # Add CIDEr scores
    cider_scores = metrics.compute_cider_for_topic(
        Path(ground_truth_path), Path(prediction_path)
    )
    results["aggregated_metrics"]["cider"] = cider_scores

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    return results
