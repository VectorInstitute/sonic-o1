"""
Main Evaluation Orchestrator
Coordinates inference and metrics computation for model evaluation
"""
import argparse
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Optional
from utils.config_loader import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_inference(
    model: str,
    tasks: List[str],
    topics: List[str],
    dataset_path: str,
    vqa_path: str,
    output_path: str,
    config_path: str,
    skip_existing: bool = True,
    experiment_name: Optional[str] = None,  
    additional_args: List[str] = None
) -> int:
    """
    Run inference for a model
    
    Args:
        model: Model name
        tasks: List of tasks (t1, t2, t3)
        topics: List of topics to evaluate
        dataset_path: Path to dataset
        vqa_path: Path to VQA ground truth
        output_path: Path to save predictions
        config_path: Path to models config
        skip_existing: Skip already processed entries
        additional_args: Additional arguments to pass
        
    Returns:
        Return code (0 for success)
    """
    logger.info(f"=" * 80)
    logger.info(f"STEP 1: Running inference for model '{model}'")
    if experiment_name: 
        logger.info(f"Experiment: {experiment_name}")

    logger.info(f"=" * 80)
    
    task_mapping = {
        "t1": "task1_summarization",
        "t2": "task2_mcq",
        "t3": "task3_temporal_localization"
    }
    full_task_names = [task_mapping.get(t, t) for t in tasks]
    
    cmd = [
        sys.executable,
        "inference/run_inference.py",
        "--model", model,
        "--tasks", *full_task_names,  # Use full names
        "--topics", *topics,
        "--config", config_path,
    ]
    if experiment_name:
        cmd.extend(["--experiment-name", experiment_name])
    

    if not skip_existing:
        cmd.append("--overwrite")

    if additional_args:
        cmd.extend(additional_args)
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error(f"Inference failed with return code {result.returncode}")
        return result.returncode
    
    logger.info(f"Inference completed successfully for {model}")
    return 0

def run_metrics(
    model: str,
    tasks: List[str],
    topics: List[str],
    config_path: str,
    use_llm_judge: bool = True,
    experiment_name: Optional[str] = None 
) -> int:
    """
    Compute metrics for a model
    
    Args:
        model: Model name
        tasks: List of tasks
        topics: List of topics
        config_path: Path to config file
        use_llm_judge: Whether to use LLM judge
        
    Returns:
        Return code (0 for success)
    """
    logger.info(f"=" * 80)
    logger.info(f"STEP 2: Computing metrics for model '{model}'")
    if experiment_name: 
        logger.info(f"Experiment: {experiment_name}")

    logger.info(f"=" * 80)
    
    cmd = [
        sys.executable,
        "metrics/compute_metrics.py",
        "--model", model,
        "--tasks", *tasks,
        "--topics", *topics,
        "--config", config_path,
    ]
    if experiment_name:
        cmd += ["--experiment-name", experiment_name]

    if not use_llm_judge:
        cmd.append("--no-llm-judge")
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error(f"Metrics computation failed with return code {result.returncode}")
        return result.returncode
    
    logger.info(f"✓ Metrics computed successfully for {model}")
    return 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluation Pipeline - Run inference and compute metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                    # Run full evaluation for one model
                    python run_evaluation.py --model gemini --tasks all
                    
                    # Run inference only
                    python run_evaluation.py --model gemini --inference-only
                    
                    # Run metrics only (after inference is done)
                    python run_evaluation.py --model gemini --metrics-only
                    
                    # Evaluate specific tasks
                    python run_evaluation.py --model gpt4o --tasks t1 t2
                    
                    # Evaluate multiple models
                    python run_evaluation.py --models gemini gpt4o qwen --tasks all
                    
                    # Skip LLM judge for faster evaluation
                    python run_evaluation.py --model gemini --no-llm-judge
                        """
    )
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        type=str,
        help="Single model name to evaluate"
    )
    model_group.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Multiple model names to evaluate"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models_config.yaml",
        help="Path to models configuration file"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only failed entries during inference"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name for organizing results (e.g., 'modality_audio_only', 'frames_16')"
    )
    
    # Task and topic selection
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Tasks to evaluate: t1, t2, t3, or 'all' (default: from config)"
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        help="Topics to evaluate (default: from config)"
    )
    
    # Optional path overrides (use config defaults if not specified)
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Override dataset path from config"
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
        "--scores-path",
        type=str,
        help="Override scores path from config"
    )
    
    # Pipeline control
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Only run inference, skip metrics computation"
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only compute metrics, skip inference"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already processed entries during inference (default: True)"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-run even if outputs exist"
    )
    
    # Evaluation options
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM judge evaluation (faster but less comprehensive)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Get all values from config with optional CLI overrides
    tasks = args.tasks if args.tasks else None
    if tasks and "all" in tasks:
        tasks = ["t1", "t2", "t3"]
    elif not tasks:
        # Get from config and convert to short form
        config_tasks = config.get("tasks", [])
        tasks = [f"t{i}" for i in range(1, len(config_tasks) + 1)]
    
    topics = args.topics if args.topics else config.get_topics()
    dataset_path = args.dataset_path if args.dataset_path else config.get_dataset_path()
    vqa_path = args.vqa_path if args.vqa_path else config.get_vqa_path()
    predictions_path = args.predictions_path if args.predictions_path else config.get("results.predictions_path", "results/predictions")
    scores_path = args.scores_path if args.scores_path else config.get("results.scores_path", "results/scores")
    additional_args = ["--retry-failed"] if args.retry_failed else None
    # Get list of models to evaluate
    models_to_evaluate = []
    if args.models:
        models_to_evaluate = args.models
    elif args.model:
        models_to_evaluate = [args.model]
    
    logger.info("=" * 80)
    logger.info("EVALUATION PIPELINE STARTING")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Models: {models_to_evaluate}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Topics: {len(topics)} topics")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"VQA: {vqa_path}")
    logger.info(f"Predictions: {predictions_path}")
    logger.info(f"Scores: {scores_path}")
    logger.info(f"LLM Judge: {'disabled' if args.no_llm_judge else 'enabled'}")
    
    if args.inference_only and args.metrics_only:
        logger.error("Cannot specify both --inference-only and --metrics-only")
        return 1
    
    skip_existing = args.skip_existing and not args.force_rerun
    
    # Evaluate each model
    failed_models = []
    
    for i, model_name in enumerate(models_to_evaluate, 1):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"EVALUATING MODEL {i}/{len(models_to_evaluate)}: {model_name}")
        if args.experiment_name: 
            logger.info(f"Experiment: {args.experiment_name}")

        logger.info("=" * 80)
        
        try:
            # Step 1: Run inference (unless metrics-only)
            
            if not args.metrics_only:
                result = run_inference(
                    model=model_name,
                    tasks=tasks,
                    topics=topics,
                    dataset_path=dataset_path,
                    vqa_path=vqa_path,
                    output_path=predictions_path,
                    config_path=args.config,
                    skip_existing=skip_existing,
                    experiment_name=args.experiment_name,
                    additional_args=additional_args
                )
                
                if result != 0:
                    logger.error(f"Inference failed for {model_name}")
                    failed_models.append((model_name, "inference"))
                    continue
            
            # Step 2: Compute metrics (unless inference-only)
            if not args.inference_only:
                result = run_metrics(
                    model=model_name,
                    tasks=tasks,
                    topics=topics,
                    config_path=args.config,
                    use_llm_judge=not args.no_llm_judge,
                    experiment_name=args.experiment_name  
                )
                
                if result != 0:
                    logger.error(f"Metrics computation failed for {model_name}")
                    failed_models.append((model_name, "metrics"))
                    continue
            
            logger.info(f"✓ Successfully completed evaluation for {model_name}")
        
        except Exception as e:
            logger.error(f"Unexpected error evaluating {model_name}: {e}", exc_info=True)
            failed_models.append((model_name, "unknown"))
            continue
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATION PIPELINE COMPLETED")
    logger.info("=" * 80)
    
    successful = len(models_to_evaluate) - len(failed_models)
    logger.info(f"Successful: {successful}/{len(models_to_evaluate)}")
    
    if failed_models:
        logger.info(f"Failed: {len(failed_models)}/{len(models_to_evaluate)}")
        for model_name, stage in failed_models:
            logger.info(f"  - {model_name} (failed at {stage})")
        return 1
    
    logger.info("All models evaluated successfully!")
    
    # Print next steps
    if not args.inference_only and not args.metrics_only:
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  - View results in: {scores_path}")
        logger.info(f"  - Generate visualizations: python visualization/plot_results.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())