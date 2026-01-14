"""
LLM-as-Judge Implementation using GPT-5-mini
Evaluates semantic similarity, factual correctness, and completeness
"""
from openai import OpenAI
import json
from typing import Dict, Any, Optional
import logging
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in the evaluation_Inference directory
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded environment variables from {env_path}")
except ImportError:
    # dotenv not installed, rely on system environment variables
    pass

logger = logging.getLogger(__name__)

class LLMJudge:
    """LLM-as-Judge evaluator using GPT-5-mini"""
    
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

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini"):
        """
        Initialize LLM Judge
        
        Args:
            api_key: OpenAI API key (if None, uses env variable)
            model: Model to use for judging (default: gpt-5-mini)
        """
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        
    def evaluate(
        self, 
        question: str, 
        correct_answer: str, 
        predicted_answer: str,
        task_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Evaluate a predicted answer against ground truth
        
        Args:
            question: The question being answered
            correct_answer: Ground truth answer
            predicted_answer: Model's predicted answer
            task_type: Type of task (for logging purposes)
            
        Returns:
            Dict with 'score' (0-10) and 'justification' (str)
        """
        user_prompt = f"""Please evaluate the following video-based question-answer pair:

                        **Question:** {question}

                        **Correct Answer:** {correct_answer}

                        **Predicted Answer:** {predicted_answer}

                        Please return your evaluation in the specified JSON format with both a score and a justification."""

        try:
            # Use the new Responses API for GPT-5-mini
            result = self.client.responses.create(
                model=self.model,
                input=f"{self.SYSTEM_PROMPT}\n\n{user_prompt}",
                reasoning={"effort": "medium"},  # Balance between speed and accuracy
                text={"verbosity": "low"},
            )
            
            # Parse the JSON response
            response_text = result.output_text.strip()
            
            
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
                            

                from json_repair import repair_json
                repaired_json = repair_json(response_text, return_objects=False)
                response_text = json.loads(repaired_json)

        
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
                "justification": f"Error parsing LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            return {
                "score": 0,
                "justification": f"Evaluation error: {str(e)}"
            }
    
    def batch_evaluate(
        self,
        evaluations: list[Dict[str, str]],
        task_type: str = "general"
    ) -> list[Dict[str, Any]]:
        """
        Evaluate multiple question-answer pairs
        
        Args:
            evaluations: List of dicts with 'question', 'correct_answer', 'predicted_answer'
            task_type: Type of task
            
        Returns:
            List of evaluation results
        """
        results = []
        for i, eval_item in enumerate(evaluations):
            logger.info(f"Evaluating {i+1}/{len(evaluations)}")
            result = self.evaluate(
                question=eval_item["question"],
                correct_answer=eval_item["correct_answer"],
                predicted_answer=eval_item["predicted_answer"],
                task_type=task_type
            )
            results.append(result)
        
        return results


# Convenience function
def evaluate_with_llm_judge(
    question: str,
    correct_answer: str,
    predicted_answer: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Convenience function for single evaluation
    
    Args:
        question: The question
        correct_answer: Ground truth
        predicted_answer: Model prediction
        api_key: OpenAI API key
        model: Model to use
        
    Returns:
        Dict with score and justification
    """
    judge = LLMJudge(api_key=api_key, model=model)
    return judge.evaluate(question, correct_answer, predicted_answer)