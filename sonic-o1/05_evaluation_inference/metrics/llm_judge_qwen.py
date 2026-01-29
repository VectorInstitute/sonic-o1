"""llm_judge_qwen.py

LLM-as-Judge using Qwen3: text-only evaluation with multi-GPU support.

Author: SONIC-O1 Team
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


def load_config(config_path: str = "models_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file; search parent dirs if path not found."""
    config_file = Path(config_path)

    # Try to find config in common locations
    search_paths = [
        config_file,
        Path(__file__).parent / config_path,
        Path(__file__).parent.parent / config_path,
    ]

    for path in search_paths:
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)

    logger.warning("Config file not found, using defaults")
    return {}


class LLMJudge:
    """LLM-as-Judge using Qwen3-8B for text-only evaluation (multi-GPU)."""

    SYSTEM_PROMPT = """You are an intelligent and fair evaluator AI that specializes in assessing the correctness and semantic alignment between ground truth answers and predicted responses for question-answering tasks, including those based on video content.

                        Your role is to evaluate how well a predicted answer matches the correct (reference) answer based on the following detailed criteria:

                        ## EVALUATION INSTRUCTIONS

                        - Focus on **semantic similarity**, **factual correctness**, and **completeness**.
                        - Accept paraphrases, synonyms, or rephrasings **as valid**, as long as they preserve the original meaning.
                        - **Do not penalize** for stylistic differences or changes in tone, unless they impact factual accuracy.
                        - **Penalize** if:
                        - The predicted answer omits **key factual elements** present in the correct answer.
                        - The prediction includes **hallucinated content** or unfounded details.
                        - The prediction **contradicts** the correct answer.
                        - Use human-like judgment: apply reasoning beyond surface text similarity.
                        - When uncertain, provide a **conservative but fair** score.
                        - Use a scoring scale from **0 (completely incorrect)** to **10 (perfect match)**.

                        ## OUTPUT FORMAT

                        Return a JSON object with **two fields**:
                        - "score": an integer from 0 to 10
                        - "justification": a concise explanation (1-3 sentences) of your reasoning

                        **Example Output:**
                        ```json
                        {
                        "score": 7,
                        "justification": "The predicted answer captures the main idea, but it omits some key details about the setting described in the correct answer."
                        }
                        ```

                        Be fair, consistent, and concise. Follow the format exactly."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device_map: Optional[str] = None,
        dtype: Optional[str] = None,
        max_memory: Optional[Dict[int, str]] = None,
        config_path: str = "models_config.yaml",
    ):
        """
        Initialize LLM Judge with Qwen3-8B.

        Args:
            model_path: HuggingFace model path (if None, reads from config)
            device_map: Device mapping strategy (if None, reads from config)
            dtype: torch dtype as string (if None, reads from config)
            max_memory: Optional dict mapping device id to memory limit
            config_path: Path to models_config.yaml
        """
        # Load config
        config = load_config(config_path)
        metrics_config = config.get("metrics", {})

        # Get model parameters from config or use provided/defaults
        self.model_path = model_path or metrics_config.get(
            "llm_judge_model", "Qwen/Qwen3-8B"
        )

        # Device map
        if device_map is None:
            device_map = metrics_config.get("llm_judge_device_map", "auto")
        self.device_map = device_map

        # Dtype
        if dtype is None:
            dtype_str = metrics_config.get("llm_judge_dtype", "bfloat16")
        else:
            dtype_str = dtype

        # Convert string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": None,
        }
        self.dtype = dtype_map.get(dtype_str, torch.bfloat16)

        # Max memory
        if max_memory is None:
            max_memory = metrics_config.get("llm_judge_max_memory")
        self.max_memory = max_memory

        # Generation config
        gen_config = metrics_config.get("llm_judge_generation", {})
        self.temperature = gen_config.get("temperature", 0.0)
        self.top_p = gen_config.get("top_p", 0.95)
        self.max_new_tokens = gen_config.get("max_new_tokens", 512)

        self.model = None
        self.tokenizer = None

        # Log configuration
        logger.info("LLM Judge Configuration:")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Device map: {self.device_map}")
        logger.info(f"  Dtype: {dtype_str}")
        logger.info(f"  Temperature: {self.temperature}")

        # Get GPU info
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"  Available GPUs: {num_gpus}")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                logger.info(f"    GPU {i}: {props.name} ({memory_gb:.1f} GB)")

    def load(self) -> None:
        """Load model and tokenizer with multi-GPU support."""
        try:
            logger.info(f"Loading LLM Judge from {self.model_path}...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with device_map for multi-GPU distribution
            logger.info("Loading model with multi-GPU distribution...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                trust_remote_code=True,
                max_memory=self.max_memory,
                offload_folder="offload",  # Fallback to disk if needed
            )

            self.model.eval()

            # Log device map
            if hasattr(self.model, "hf_device_map"):
                logger.info(f"Model device map: {self.model.hf_device_map}")

            logger.info("LLM Judge loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load LLM Judge model: {e}") from e

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

        gc.collect()

        logger.info("LLM Judge unloaded from all GPUs")

    def evaluate(
        self,
        question: str,
        correct_answer: str,
        predicted_answer: str,
        task_type: str = "general",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a predicted answer against ground truth.

        Args:
            question: The question being answered
            correct_answer: Ground truth answer
            predicted_answer: Model's predicted answer
            task_type: Type of task (for logging purposes)
            temperature: Generation temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            max_new_tokens: Maximum tokens to generate (uses config default if None)

        Returns:
            Dict with "score" (0-10) and "justification" (str).
        """
        # Lazy load if needed
        if self.model is None or self.tokenizer is None:
            logger.info("Model not loaded, loading now...")
            self.load()

        # Use config defaults if not provided
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Construct evaluation prompt
        user_prompt = f"""Please evaluate the following video-based question-answer pair:

                    **Question:** {question}

                    **Correct Answer:** {correct_answer}

                    **Predicted Answer:** {predicted_answer}

                    Please return your evaluation in the specified JSON format with both a score and a justification."""

        # Build conversation using Qwen3 chat template
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # Apply chat template
            text_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            # Tokenize
            inputs = self.tokenizer(
                text_prompt, return_tensors="pt", padding=True, truncation=True
            )

            # Move inputs to first device (accelerate handles the rest)
            if hasattr(self.model, "hf_device_map"):
                first_device = next(iter(self.model.hf_device_map.values()))
            else:
                first_device = next(self.model.parameters()).device

            inputs = {k: v.to(first_device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    top_p=top_p if temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response (only the new tokens)
            generated_ids = outputs[0][inputs["input_ids"].shape[1] :].tolist()

            # Parse thinking content using token ID
            try:
                # Find </think> token (ID: 151668)
                index = len(generated_ids) - generated_ids[::-1].index(151668)
            except ValueError:
                # No thinking token found
                index = 0

            # Decode only the content after </think>
            response_text = self.tokenizer.decode(
                generated_ids[index:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            # Parse JSON response
            return self._parse_response(response_text)
        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}", exc_info=True)
            return {"score": 0, "justification": f"Evaluation error: {str(e)}"}

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON response from model output.

        Args:
            response_text: Raw model output

        Returns:
            Dict with "score" and "justification".
        """
        try:
            # Handle thinking blocks
            if "</think>" in response_text:
                response_text = response_text.split("</think>")[-1].strip()

            # Try to extract JSON from response
            if "```json" in response_text:
                # Extract JSON from markdown code block
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                # Extract from generic code block
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            # Try to find JSON object with curly braces
            if "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                response_text = response_text[json_start:json_end]

            evaluation = json.loads(response_text)

            # Validate structure
            if "score" not in evaluation or "justification" not in evaluation:
                raise ValueError("Missing required fields in LLM response")

            # Ensure score is integer 0-10
            evaluation["score"] = max(0, min(10, int(evaluation["score"])))

            return evaluation

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM judge response: {e}")
            logger.error(f"Raw response: {response_text}")
            return {
                "score": 0,
                "justification": f"Error parsing LLM response: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return {"score": 0, "justification": f"Response processing error: {str(e)}"}

    def batch_evaluate(
        self,
        evaluations: list[Dict[str, str]],
        task_type: str = "general",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> list[Dict[str, Any]]:
        """
        Evaluate multiple question-answer pairs.

        Args:
            evaluations: List of dicts with question, correct_answer, predicted_answer
            task_type: Type of task
            temperature: Generation temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            max_new_tokens: Max tokens per generation (uses config default if None)

        Returns:
            List of evaluation results (score and justification per item).
        """
        results = []

        # Ensure model is loaded
        if self.model is None:
            self.load()

        for i, eval_item in enumerate(evaluations):
            logger.info(f"Evaluating {i + 1}/{len(evaluations)}")
            result = self.evaluate(
                question=eval_item["question"],
                correct_answer=eval_item["correct_answer"],
                predicted_answer=eval_item["predicted_answer"],
                task_type=task_type,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            results.append(result)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_path": self.model_path,
            "dtype": str(self.dtype),
            "device_map": self.device_map,
        }

        if self.model is not None and hasattr(self.model, "hf_device_map"):
            info["loaded_device_map"] = self.model.hf_device_map

        if torch.cuda.is_available():
            info["num_gpus"] = torch.cuda.device_count()
            info["gpu_memory_allocated"] = {
                i: f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB"
                for i in range(torch.cuda.device_count())
            }

        return info


# Convenience function for backward compatibility
def evaluate_with_llm_judge(
    question: str,
    correct_answer: str,
    predicted_answer: str,
    model_path: Optional[str] = None,
    device_map: Optional[str] = None,
    dtype: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate single instance.

    Args:
        question: The question
        correct_answer: Ground truth
        predicted_answer: Model prediction
        model_path: Path to Qwen model (reads from config if None)
        device_map: Device mapping strategy (reads from config if None)
        dtype: torch dtype string (reads from config if None)

    Returns:
        Dict with "score" and "justification".
    """
    judge = LLMJudge(model_path=model_path, device_map=device_map, dtype=dtype)
    judge.load()
    result = judge.evaluate(question, correct_answer, predicted_answer)
    judge.unload()
    return result
