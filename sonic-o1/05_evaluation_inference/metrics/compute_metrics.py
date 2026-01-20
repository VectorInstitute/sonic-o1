"""
Main Metrics Computation Script
Orchestrates evaluation for all tasks and topics
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional
import sys

from t1_metrics import evaluate_t1_topic
from t2_metrics import evaluate_t2_topic
from t3_metrics import evaluate_t3_topic

sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_existing_topic_results(output_path: Path, model_name: str, task_name: str, 
                                topics: List[str], judge_dir: str, 
                                experiment_name: Optional[str] = None) -> dict:
    """Load previously computed topic results"""
    existing_results = {}
    
    for topic in topics:
        if experiment_name:
            topic_file = output_path / judge_dir / experiment_name / model_name / task_name / f"{topic}.json"
        else:
            topic_file = output_path / judge_dir / model_name / task_name / f"{topic}.json"
        
        if topic_file.exists():
            try:
                with open(topic_file, 'r') as f:
                    data = json.load(f)
                    existing_results[topic] = data.get("aggregated_metrics", {})
                    logger.info(f"    Loaded existing results for {topic}")
            except Exception as e:
                logger.warning(f"    Failed to load {topic_file}: {e}")
    
    return existing_results

def get_task_mapping(config):
    """Get task mapping from config"""
    tasks = config.get("tasks", [])
    mapping = {}
    for i, task in enumerate(tasks, 1):
        mapping[f"t{i}"] = task
    return mapping


def compute_metrics_for_model(
    model_name: str,
    tasks: List[str],
    topics: List[str],
    vqa_path: Path,
    predictions_path: Path,
    output_path: Path,
    use_llm_judge: bool = True,
    config = None,
    experiment_name: Optional[str] = None
):
    """
    Compute metrics for all tasks and topics for a model
    
    Args:
        model_name: Name of the model
        tasks: List of task names to evaluate
        topics: List of topics to evaluate
        vqa_path: Path to VQA ground truth directory
        predictions_path: Path to predictions directory
        output_path: Path to save results
        use_llm_judge: Whether to use LLM judge
    """
    logger.info(f"Computing metrics for model: {model_name}")
    task_mapping = get_task_mapping(config)
    judge_name = config.get_llm_judge_model()
    logger.info(f"Using LLM judge: {judge_name} (enabled: {use_llm_judge})")

    if 'gpt' in judge_name.lower():
        judge_name = 'gpt'
        judge_dir = 'gpt_judge'
    elif 'qwen' in judge_name.lower():
        judge_dir = 'qwen_judge'
        judge_name= 'qwen'
    else:
        raise ValueError(f"Unknown judge name: {judge_name}")

    results = {
        "model": model_name,
        "experiment_name": experiment_name, 
        "tasks": {}
    }

    
    for task_key in tasks:
        task_name = task_mapping[task_key]
        logger.info(f"Evaluating task: {task_name}")
        
        task_results = {
            "task_name": task_name,
            "topics": {}
        }
        all_topics_for_aggregation = config.get_topics()  # Get all possible topics
        existing_results = load_existing_topic_results(
            output_path, model_name, task_name, 
            all_topics_for_aggregation, judge_dir, experiment_name
        )
        task_results["topics"] = existing_results  # Start with existing
    
        
        for topic in topics:
            logger.info(f"  Processing topic: {topic}")
            
            # Paths
            gt_path = vqa_path / task_name / f"{topic}.json"
            if experiment_name:
                pred_path = predictions_path / experiment_name / model_name / task_name / f"{topic}.json"
            else:
                pred_path = predictions_path / model_name / task_name / f"{topic}.json"
            
            
            # Check if files exist
            if not gt_path.exists():
                logger.warning(f"Ground truth not found: {gt_path}")
                continue
            
            if not pred_path.exists():
                logger.warning(f"Prediction not found: {pred_path}")
                continue
            
            # Compute task-specific metrics
            if experiment_name:
                topic_output_path = output_path  /judge_dir/ experiment_name/ model_name / task_name / f"{topic}.json"
            else:
                topic_output_path = output_path /judge_dir/ model_name / task_name / f"{topic}.json"
            
            logger.info(f"    Saving topic results to: {topic_output_path}")

            topic_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if task_key == "t1":
                    topic_results = evaluate_t1_topic(
                        str(gt_path),
                        str(pred_path),
                        str(topic_output_path),
                        use_llm_judge=use_llm_judge,
                        judge_name= judge_name
                    )
                elif task_key == "t2":
                    topic_results = evaluate_t2_topic(
                        str(gt_path),
                        str(pred_path),
                        str(topic_output_path),
                        use_llm_judge=use_llm_judge,
                        judge_name= judge_name
                    )
                elif task_key == "t3":
                    topic_results = evaluate_t3_topic(
                        str(gt_path),
                        str(pred_path),
                        str(topic_output_path),
                        use_llm_judge=use_llm_judge,
                        judge_name= judge_name
                    )
                
                task_results["topics"][topic] = topic_results["aggregated_metrics"]
            
            except Exception as e:
                logger.error(f"Failed to evaluate {topic} for task {task_name}: {e}", exc_info=True)
                continue
        
        # Aggregate across topics
        task_results["aggregated_across_topics"] = aggregate_topic_metrics(task_results["topics"], task_key)
        
        results["tasks"][task_key] = task_results
    
    # Save overall results
    if experiment_name:
        overall_output_path = output_path / judge_dir/experiment_name /model_name / "overall_metrics.json"
    else:
        overall_output_path = output_path / judge_dir /model_name / "overall_metrics.json"
    
    overall_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(overall_output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Overall metrics saved to {overall_output_path}")
    
    return results


def aggregate_topic_metrics(topic_metrics: dict, task_key: str) -> dict:
    """Aggregate metrics across all topics for a task"""
    if not topic_metrics:
        return {}
    
    aggregated = {}
    
    if task_key == "t1":
        # Aggregate T1 metrics
        for summary_type in ["detailed", "short"]:
            rouge_scores = []
            sim_scores = []
            llm_scores = []
            
            for topic, metrics in topic_metrics.items():
                if summary_type in metrics:
                    rouge_scores.append(metrics[summary_type]["rouge_l_mean"])
                    sim_scores.append(metrics[summary_type]["text_similarity_mean"])
                    if "llm_judge_score_mean" in metrics[summary_type]:
                        llm_scores.append(metrics[summary_type]["llm_judge_score_mean"])
            
            aggregated[summary_type] = {
                "rouge_l_mean": float(sum(rouge_scores) / len(rouge_scores)) if rouge_scores else 0.0,
                "text_similarity_mean": float(sum(sim_scores) / len(sim_scores)) if sim_scores else 0.0,
            }
            
            if llm_scores:
                aggregated[summary_type]["llm_judge_score_mean"] = float(sum(llm_scores) / len(llm_scores))
        
        # Aggregate CIDEr
        cider_detailed = []
        cider_short = []
        for topic, metrics in topic_metrics.items():
            if "cider" in metrics:
                cider_detailed.append(metrics["cider"]["cider_detailed"])
                cider_short.append(metrics["cider"]["cider_short"])
        
        if cider_detailed:
            aggregated["cider"] = {
                "cider_detailed_mean": float(sum(cider_detailed) / len(cider_detailed)),
                "cider_short_mean": float(sum(cider_short) / len(cider_short))
            }
    
    elif task_key == "t2":
        # Aggregate T2 metrics
        accuracies = []
        rouge_scores = []
        sim_scores = []
        llm_scores = []
        
        for topic, metrics in topic_metrics.items():
            accuracies.append(metrics["accuracy"])
            
            if "rationale" in metrics:
                rouge_scores.append(metrics["rationale"]["rouge_l_mean"])
                sim_scores.append(metrics["rationale"]["text_similarity_mean"])
                if "llm_judge_score_mean" in metrics["rationale"]:
                    llm_scores.append(metrics["rationale"]["llm_judge_score_mean"])
        
        aggregated["accuracy_mean"] = float(sum(accuracies) / len(accuracies)) if accuracies else 0.0
        
        if rouge_scores:
            aggregated["rationale"] = {
                "rouge_l_mean": float(sum(rouge_scores) / len(rouge_scores)),
                "text_similarity_mean": float(sum(sim_scores) / len(sim_scores)),
            }
            
            if llm_scores:
                aggregated["rationale"]["llm_judge_score_mean"] = float(sum(llm_scores) / len(llm_scores))
    
    elif task_key == "t3":
        # Aggregate T3 metrics
        mean_ious = []
        mae_avgs = []
        rouge_scores = []
        sim_scores = []
        llm_scores = []
        recall_metrics = {0.3: [], 0.5: [], 0.7: []}
        
        for topic, metrics in topic_metrics.items():
            mean_ious.append(metrics["mean_iou"])
            mae_avgs.append(metrics["mae"]["average_mean"])
            
            for threshold in [0.3, 0.5, 0.7]:
                key = f"R@{threshold}"
                if key in metrics:
                    recall_metrics[threshold].append(metrics[key]["recall"])
            
            if "rationale" in metrics:
                rouge_scores.append(metrics["rationale"]["rouge_l_mean"])
                sim_scores.append(metrics["rationale"]["text_similarity_mean"])
                if "llm_judge_score_mean" in metrics["rationale"]:
                    llm_scores.append(metrics["rationale"]["llm_judge_score_mean"])
        
        aggregated["mean_iou"] = float(sum(mean_ious) / len(mean_ious)) if mean_ious else 0.0
        aggregated["mae_average"] = float(sum(mae_avgs) / len(mae_avgs)) if mae_avgs else 0.0
        
        for threshold, recalls in recall_metrics.items():
            if recalls:
                aggregated[f"R@{threshold}"] = float(sum(recalls) / len(recalls))
        
        if rouge_scores:
            aggregated["rationale"] = {
                "rouge_l_mean": float(sum(rouge_scores) / len(rouge_scores)),
                "text_similarity_mean": float(sum(sim_scores) / len(sim_scores)),
            }
            
            if llm_scores:
                aggregated["rationale"]["llm_judge_score_mean"] = float(sum(llm_scores) / len(llm_scores))
    
    return aggregated

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to evaluate"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Multiple model names to evaluate"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all models in predictions directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name (must match inference experiment name)"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=["t1", "t2", "t3"],
        help="Tasks to evaluate (default: from config or all tasks)"
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        help="Topics to evaluate (default: from config or all topics)"
    )
    parser.add_argument(
        "--vqa-path",
        type=str,
        help="Override VQA path from config"
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        help="Override predictions path from config"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Override output path from config"
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM judge evaluation (faster)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    
    # Get all values from config with optional CLI overrides
    tasks = args.tasks if args.tasks else ["t1", "t2", "t3"]
    topics = args.topics if args.topics else config.get_topics()
    vqa_path = args.vqa_path if args.vqa_path else config.get_vqa_path()
    predictions_path = args.predictions_path if args.predictions_path else config.get("results.predictions_path", "results/predictions")
    output_path = args.output_path if args.output_path else config.get("results.scores_path", "results/scores")
    
    # Determine which models to evaluate
    models_to_evaluate = []
    
    if args.all:
        predictions_path_obj = Path(predictions_path)
        if args.experiment_name:
            predictions_path_obj = predictions_path_obj / args.experiment_name
        
        if predictions_path_obj.exists():
            models_to_evaluate = [
                d.name for d in predictions_path_obj.iterdir() 
                if d.is_dir() and not d.name.startswith(".")
            ]
        else:
            logger.error(f"Predictions path not found: {predictions_path_obj}")
            return 1
    elif args.models:
        models_to_evaluate = args.models
    elif args.model:
        models_to_evaluate = [args.model]
    else:
        logger.error("Must specify --model, --models, or --all")
        return 1
    
    logger.info(f"Evaluating models: {models_to_evaluate}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Topics: {len(topics)} topics")
    if args.experiment_name: 
        logger.info(f"Experiment: {args.experiment_name}")

    logger.info(f"VQA path: {vqa_path}")
    logger.info(f"Predictions path: {predictions_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"LLM Judge: {'disabled' if args.no_llm_judge else 'enabled'}")
    
    # Evaluate each model
    for model_name in models_to_evaluate:
        try:
            compute_metrics_for_model(
                model_name=model_name,
                tasks=tasks,
                topics=topics,
                vqa_path=Path(vqa_path),
                predictions_path=Path(predictions_path),
                output_path=Path(output_path),
                use_llm_judge=not args.no_llm_judge,
                config=config,
                experiment_name=args.experiment_name  
            )
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_name}: {e}", exc_info=True)
            continue
    
    logger.info("Metrics computation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())