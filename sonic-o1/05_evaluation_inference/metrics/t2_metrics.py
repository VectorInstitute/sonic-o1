"""
T2 Metrics: Question Answering (MCQ) Evaluation
Computes accuracy, ROUGE-L, CIDEr, text similarity, and LLM-as-Judge for rationales
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def make_key(entry):
    seg = entry["segment"]
    return (entry['video_id'], entry['video_number'], float(seg['start']), float(seg['end']))

class T2Metrics:
    """Compute metrics for T2: Question Answering (MCQ)"""
    
    def __init__(
        self,
        use_llm_judge: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        judge_name: str = "gpt"
    ):
        """
        Initialize T2 metrics
        
        Args:
            use_llm_judge: Whether to use LLM judge for rationale evaluation
            embedding_model: Model for computing text similarity
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.cider_scorer = Cider()
        self.embedding_model = SentenceTransformer(embedding_model)
        
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
    
    def compute_accuracy(self, ground_truth_answer: str, predicted_answer: str) -> int:
        """
        Compute exact match accuracy
        
        Args:
            ground_truth_answer: Ground truth answer (letter or index)
            predicted_answer: Predicted answer
            
        Returns:
            1 if correct, 0 if incorrect
        """
        # Normalize answers (handle both letter and index format)
        gt_normalized = str(ground_truth_answer).strip().upper()
        pred_normalized = str(predicted_answer).strip().upper()
        
        return 1 if gt_normalized == pred_normalized else 0
    
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
    
    def evaluate_entry(
        self,
        ground_truth: Dict[str, Any],
        prediction: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single question entry
        
        Args:
            ground_truth: Ground truth entry
            prediction: Predicted entry
            
        Returns:
            Dict with all metrics, or None if prediction failed
        """
        # Skip failed predictions
        if "error" in prediction or "outputs" not in prediction:
            logger.warning(
                f"Skipping failed entry: video={prediction.get('video_id', 'unknown')}, "
                f"segment={prediction.get('segment', 'unknown')}"
            )
            return None
        
        results = {
            "video_id": ground_truth["video_id"],
            "video_number": ground_truth["video_number"],
            "segment": ground_truth["segment"],
            "question": ground_truth["question"]
        }
        
        # Answer accuracy
        gt_answer_letter = ground_truth.get("answer_letter", "")
        gt_answer_index = ground_truth.get("answer_index", -1)
        
        pred_answer_letter = prediction["outputs"].get("answer_letter", "")
        pred_answer_index = prediction["outputs"].get("answer_index", -1)
        
        # Check both letter and index
        accuracy_letter = self.compute_accuracy(gt_answer_letter, pred_answer_letter)
        accuracy_index = self.compute_accuracy(gt_answer_index, pred_answer_index)
        
        results["answer_correct"] = max(accuracy_letter, accuracy_index)  # Either format matches
        results["gt_answer"] = gt_answer_letter
        results["pred_answer"] = pred_answer_letter
        
        # Rationale evaluation
        gt_rationale = ground_truth.get("rationale", "")
        pred_rationale = prediction["outputs"].get("rationale", "")
        
        if gt_rationale and pred_rationale:
            results["rationale_metrics"] = {
                "rouge_l": self.compute_rouge_l(gt_rationale, pred_rationale),
                "text_similarity": self.compute_text_similarity(gt_rationale, pred_rationale)
            }
            
            # LLM Judge for rationale quality
            if self.use_llm_judge:
                try:
                    llm_eval = self.llm_judge.evaluate(
                        question=ground_truth["question"],
                        correct_answer=gt_rationale,
                        predicted_answer=pred_rationale,
                        task_type="rationale"
                    )
                    results["rationale_metrics"]["llm_judge_score"] = llm_eval["score"]
                    results["rationale_metrics"]["llm_judge_justification"] = llm_eval["justification"]
                except Exception as e:
                    logger.error(f"LLM judge failed for question: {e}")
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
        Evaluate all questions for a topic
        
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
        
        # Evaluate each entry
        entry_results = []
        for key, gt_entry in gt_entries.items():
            if key not in pred_entries:
                logger.warning(f"Missing prediction for question: {key}")
                continue
            
            pred_entry = pred_entries[key]
            result = self.evaluate_entry(gt_entry, pred_entry)
            
            # Skip None results from failed entries
            if result is not None:
                entry_results.append(result)
        
        # Aggregate metrics
        aggregated = self._aggregate_results(entry_results)
        
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
            "num_evaluated": len(entry_results),
            "aggregated_metrics": aggregated,
            "per_entry_results": entry_results
        }
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all questions"""
        # Answer accuracy
        correct = sum(r["answer_correct"] for r in results)
        total = len(results)
        accuracy = correct / total if total > 0 else 0.0
        
        aggregated = {
            "accuracy": float(accuracy),
            "correct": correct,
            "total": total
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
                    
                    # Skip failed predictions
                    if "error" in pred_entry or "outputs" not in pred_entry:
                        continue
                    
                    gt_rat = gt_entry.get("rationale", "")
                    pred_rat = pred_entry["outputs"].get("rationale", "")
                    
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


def evaluate_t2_topic(
    ground_truth_path: str,
    prediction_path: str,
    output_path: str,
    use_llm_judge: bool = True,
    judge_name: str = "gpt",
):
    """
    Convenience function to evaluate a single topic
    
    Args:
        ground_truth_path: Path to ground truth JSON
        prediction_path: Path to prediction JSON
        output_path: Where to save results
        use_llm_judge: Whether to use LLM judge
    """
    logger.info(f"Evaluating T2: {Path(ground_truth_path).stem}")
    
    metrics = T2Metrics(use_llm_judge=use_llm_judge,judge_name=judge_name)
    results = metrics.evaluate_topic(
        Path(ground_truth_path),
        Path(prediction_path)
    )
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Accuracy: {results['aggregated_metrics']['accuracy']:.2%}")
    
    return results