"""
T3 Metrics: Temporal Localization Evaluation
Computes IoU, Mean IoU, Recall@θ, MAE, and LLM-as-Judge for rationales
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

def make_key(entry):
    seg = entry["segment"]
    return (entry['video_id'], entry['video_number'], float(seg['start']), float(seg['end']))

class T3Metrics:
    """Compute metrics for T3: Temporal Localization"""
    
    def __init__(
        self,
        use_llm_judge: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        iou_thresholds: List[float] = [0.3, 0.5, 0.7],
        judge_name: str = "gpt"
    ):
        """
        Initialize T3 metrics
        
        Args:
            use_llm_judge: Whether to use LLM judge for rationale evaluation
            embedding_model: Model for computing text similarity
            iou_thresholds: IoU thresholds for Recall@θ computation
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.cider_scorer = Cider()
        self.embedding_model = SentenceTransformer(embedding_model)
        self.iou_thresholds = iou_thresholds
        
        self.use_llm_judge = use_llm_judge
        if use_llm_judge:
            if judge_name == "gpt":
                from llm_judge_gpt import LLMJudge
                self.llm_judge = LLMJudge()
            elif judge_name == "qwen":
                from llm_judge_qwen import LLMJudge
                self.llm_judge = LLMJudge()
            else:
                raise ValueError(f"Unknown judge name: {judge_name}")
            
    def compute_iou(
        self,
        gt_start: float,
        gt_end: float,
        pred_start: float,
        pred_end: float
    ) -> float:
        """
        Compute Intersection over Union (IoU) for temporal segments
        
        Args:
            gt_start: Ground truth start time (seconds)
            gt_end: Ground truth end time (seconds)
            pred_start: Predicted start time (seconds)
            pred_end: Predicted end time (seconds)
            
        Returns:
            IoU score (0-1)
        """
        # Compute intersection
        intersection_start = max(gt_start, pred_start)
        intersection_end = min(gt_end, pred_end)
        intersection = max(0, intersection_end - intersection_start)
        
        # Compute union
        union_start = min(gt_start, pred_start)
        union_end = max(gt_end, pred_end)
        union = union_end - union_start
        
        # Avoid division by zero
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return float(iou)
    
    def compute_mae(
        self,
        gt_start: float,
        gt_end: float,
        pred_start: float,
        pred_end: float
    ) -> Tuple[float, float, float]:
        """
        Compute Mean Absolute Error for start, end, and average
        
        Args:
            gt_start: Ground truth start time
            gt_end: Ground truth end time
            pred_start: Predicted start time
            pred_end: Predicted end time
            
        Returns:
            Tuple of (start_mae, end_mae, avg_mae)
        """
        start_mae = abs(gt_start - pred_start)
        end_mae = abs(gt_end - pred_end)
        avg_mae = (start_mae + end_mae) / 2.0
        
        return float(start_mae), float(end_mae), float(avg_mae)
    
    def compute_rouge_l(self, reference: str, prediction: str) -> float:
        """Compute ROUGE-L F1 score"""
        scores = self.rouge_scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure
    
    def compute_text_similarity(self, reference: str, prediction: str) -> float:
        """Compute cosine similarity between embeddings"""
        ref_embedding = self.embedding_model.encode([reference])
        pred_embedding = self.embedding_model.encode([prediction])
        
        similarity = cosine_similarity(ref_embedding, pred_embedding)[0][0]
        return float(similarity)
        
    def evaluate_question(
        self,
        ground_truth: Dict[str, Any],
        prediction: Dict[str, Any],
        segment_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single temporal localization question
        
        Args:
            ground_truth: Ground truth question entry
            prediction: Predicted question entry (the question object itself, not wrapped)
            segment_info: Segment information (video_id, etc.)
            
        Returns:
            Dict with all metrics, or None if prediction failed
        """
        results = {
            "question_id": ground_truth["question_id"],
            "question": ground_truth["question"],
            **segment_info
        }
        
        # Extract temporal bounds
        # GT: has "answer" wrapper
        gt_answer = ground_truth.get("answer", {})
        gt_start = float(gt_answer.get("start_s", 0))
        gt_end   = float(gt_answer.get("end_s", 0))

        
        # Prediction: NO "answer" wrapper - values are directly in the question object
        pred_start = float(prediction.get("start_s", 0))
        pred_end   = float(prediction.get("end_s", 0))

        
        results["gt_interval"] = {"start": gt_start, "end": gt_end}
        results["pred_interval"] = {"start": pred_start, "end": pred_end}
        
        # Compute IoU
        iou = self.compute_iou(gt_start, gt_end, pred_start, pred_end)
        results["iou"] = iou
        
        # Compute Recall@θ for each threshold
        results["recall_at_threshold"] = {}
        for threshold in self.iou_thresholds:
            results["recall_at_threshold"][f"R@{threshold}"] = 1 if iou >= threshold else 0
        
        # Compute MAE
        start_mae, end_mae, avg_mae = self.compute_mae(
            gt_start, gt_end, pred_start, pred_end
        )
        results["mae"] = {
            "start": start_mae,
            "end": end_mae,
            "average": avg_mae
        }
        
        # Rationale evaluation
        # GT: at question level
        gt_rationale = ground_truth.get("rationale_model", "")
        # Prediction: directly in question object
        pred_rationale = prediction.get("rationale_model", "")
        
        if gt_rationale and pred_rationale:
            results["rationale_metrics"] = {
                "rouge_l": self.compute_rouge_l(gt_rationale, pred_rationale),
                "text_similarity": self.compute_text_similarity(gt_rationale, pred_rationale)
            }
            
            # LLM Judge for rationale
            if self.use_llm_judge:
                try:
                    llm_eval = self.llm_judge.evaluate(
                        question=ground_truth["question"],
                        correct_answer=gt_rationale,
                        predicted_answer=pred_rationale,
                        task_type="temporal_rationale"
                    )
                    results["rationale_metrics"]["llm_judge_score"] = llm_eval["score"]
                    results["rationale_metrics"]["llm_judge_justification"] = llm_eval["justification"]
                except Exception as e:
                    logger.error(f"LLM judge failed: {e}")
                    results["rationale_metrics"]["llm_judge_score"] = None
                    results["rationale_metrics"]["llm_judge_justification"] = str(e)
        else:
            results["rationale_metrics"] = None
        
        return results
    
    def evaluate_topic(
        self,
        ground_truth_path: Path,
        prediction_path: Path
    ) -> Dict[str, Any]:
        """
        Evaluate all temporal questions for a topic
        
        Args:
            ground_truth_path: Path to ground truth JSON
            prediction_path: Path to prediction JSON
            
        Returns:
            Dict with aggregated metrics
        """
        # Load data
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
        
        with open(prediction_path, 'r') as f:
            prediction = json.load(f)
        
        # Match entries by video_id + segment
        
        gt_entries = {make_key(e): e for e in ground_truth["entries"]}
        pred_entries = {make_key(e): e for e in prediction["entries"]}
        
        # Evaluate each question
        all_question_results = []
        
        for key, gt_entry in gt_entries.items():
            if key not in pred_entries:
                logger.warning(f"Missing prediction for segment: {key}")
                continue
            
            pred_entry = pred_entries[key]
            
            # Skip failed segment predictions - CHECK FOR "outputs" NOT "questions"
            if "error" in pred_entry or "outputs" not in pred_entry:
                logger.warning(f"Skipping failed segment: {key}")
                continue
            
            segment_info = {
                "video_id": gt_entry["video_id"],
                "video_number": gt_entry["video_number"],
                "segment": gt_entry["segment"]
            }
            
            # Match questions by question_id
            gt_questions = {q["question_id"]: q for q in gt_entry.get("questions", [])}
            # GET questions from outputs
            pred_questions = {q["question_id"]: q for q in pred_entry.get("outputs", {}).get("questions", [])}
            
            for qid, gt_q in gt_questions.items():
                if qid not in pred_questions:
                    logger.warning(f"Missing prediction for question {qid}")
                    continue
                
                pred_q = pred_questions[qid]
                result = self.evaluate_question(gt_q, pred_q, segment_info)
                
                # Skip None results from failed questions
                if result is not None:
                    all_question_results.append(result)
        
        # Aggregate metrics
        aggregated = self._aggregate_results(all_question_results)
        
        # Compute CIDEr for rationales
        cider_score = self._compute_cider_rationales(
            ground_truth_path,
            prediction_path
        )
        if cider_score is not None:
            aggregated["rationale_cider"] = cider_score
        
        return {
            "topic_id": ground_truth["topic_id"],
            "topic_name": ground_truth["topic_name"],
            "num_evaluated": len(all_question_results),
            "aggregated_metrics": aggregated,
            "per_question_results": all_question_results
        }
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all questions"""
        ious = [r["iou"] for r in results]
        
        aggregated = {
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "std_iou": float(np.std(ious)) if ious else 0.0,
            "median_iou": float(np.median(ious)) if ious else 0.0
        }
        
        # Recall@θ
        for threshold in self.iou_thresholds:
            key = f"R@{threshold}"
            recalls = [r["recall_at_threshold"][key] for r in results]
            count = sum(recalls)
            total = len(recalls)
            aggregated[key] = {
                "recall": float(count / total) if total > 0 else 0.0,
                "count": count,
                "total": total
            }
        
        # MAE
        start_maes = [r["mae"]["start"] for r in results]
        end_maes = [r["mae"]["end"] for r in results]
        avg_maes = [r["mae"]["average"] for r in results]
        
        aggregated["mae"] = {
            "start_mean": float(np.mean(start_maes)) if start_maes else 0.0,
            "end_mean": float(np.mean(end_maes)) if end_maes else 0.0,
            "average_mean": float(np.mean(avg_maes)) if avg_maes else 0.0
        }
        
        # Rationale metrics
        rationale_metrics = {
            "rouge_l": [],
            "text_similarity": [],
            "llm_judge_score": []
        }
        
        for result in results:
            if result.get("rationale_metrics"):
                rm = result["rationale_metrics"]
                rationale_metrics["rouge_l"].append(rm["rouge_l"])
                rationale_metrics["text_similarity"].append(rm["text_similarity"])
                
                if rm.get("llm_judge_score") is not None:
                    rationale_metrics["llm_judge_score"].append(rm["llm_judge_score"])
        
        if rationale_metrics["rouge_l"]:
            aggregated["rationale"] = {
                "rouge_l_mean": float(np.mean(rationale_metrics["rouge_l"])),
                "rouge_l_std": float(np.std(rationale_metrics["rouge_l"])),
                "text_similarity_mean": float(np.mean(rationale_metrics["text_similarity"])),
                "text_similarity_std": float(np.std(rationale_metrics["text_similarity"])),
            }
            
            if rationale_metrics["llm_judge_score"]:
                aggregated["rationale"]["llm_judge_score_mean"] = float(np.mean(rationale_metrics["llm_judge_score"]))
                aggregated["rationale"]["llm_judge_score_std"] = float(np.std(rationale_metrics["llm_judge_score"]))
        
        return aggregated
    
    def _compute_cider_rationales(
        self,
        ground_truth_path: Path,
        prediction_path: Path
    ) -> Optional[float]:
        """Compute CIDEr score for rationales"""
        try:
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)
            
            with open(prediction_path, 'r') as f:
                prediction = json.load(f)
            
            
            gt_entries = {make_key(e): e for e in ground_truth["entries"]}
            pred_entries = {make_key(e): e for e in prediction["entries"]}
            
            gt_rationales = []
            pred_rationales = []
            
            for key, gt_entry in gt_entries.items():
                if key in pred_entries:
                    pred_entry = pred_entries[key]
                    
                    # Skip failed segment predictions
                    if "error" in pred_entry or "outputs" not in pred_entry:
                        continue
                    
                    gt_questions = {q["question_id"]: q for q in gt_entry.get("questions", [])}
                    # GET questions from outputs
                    pred_questions = {q["question_id"]: q for q in pred_entry.get("outputs", {}).get("questions", [])}
                    
                    for qid, gt_q in gt_questions.items():
                        if qid in pred_questions:
                            pred_q = pred_questions[qid]
                            
                            # rationale_model is directly in question objects
                            gt_rat = gt_q.get("rationale_model", "")
                            pred_rat = pred_q.get("rationale_model", "")
                            
                            if gt_rat and pred_rat:
                                gt_rationales.append(gt_rat)
                                pred_rationales.append(pred_rat)
            
            if gt_rationales and pred_rationales:
                gts = {i: [ref] for i, ref in enumerate(gt_rationales)}
                res = {i: [pred] for i, pred in enumerate(pred_rationales)}
                
                score, _ = self.cider_scorer.compute_score(gts, res)
                return float(score)
        
        except Exception as e:
            logger.error(f"Failed to compute CIDEr for rationales: {e}")
        
        return None


def evaluate_t3_topic(
    ground_truth_path: str,
    prediction_path: str,
    output_path: str,
    use_llm_judge: bool = True,
    iou_thresholds: List[float] = [0.3, 0.5, 0.7],
    judge_name: str = "gpt"
):
    """
    Convenience function to evaluate a single topic
    
    Args:
        ground_truth_path: Path to ground truth JSON
        prediction_path: Path to prediction JSON
        output_path: Where to save results
        use_llm_judge: Whether to use LLM judge
        iou_thresholds: IoU thresholds for recall computation
    """
    logger.info(f"Evaluating T3: {Path(ground_truth_path).stem}")
    
    metrics = T3Metrics(
        use_llm_judge=use_llm_judge,
        iou_thresholds=iou_thresholds,
        judge_name=judge_name
    )
    results = metrics.evaluate_topic(
        Path(ground_truth_path),
        Path(prediction_path)
    )
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Mean IoU: {results['aggregated_metrics']['mean_iou']:.3f}")
    
    return results