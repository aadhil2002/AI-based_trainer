import json
import os
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from argparse import ArgumentParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-7b-chat-hf"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95

class InterviewTipGenerator:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16, 
                device_map=self.device
            )
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    @staticmethod
    def load_json_file(file_path: str) -> Dict[str, Any]:
        logger.info(f"Loading JSON file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {file_path}")
            raise

    @staticmethod
    def generate_prompt(analysis_results: Dict[str, Any]) -> str:
        prompt = "Based on the following interview analysis, provide specific tips to improve the interviewee's skills:\n\n"
        
        sections = [
            ('Engagement', analysis_results.get('engagement_and_interaction', {}).get('video', [])),
            ('Confidence', analysis_results.get('confidence_level', {})),
            ('Voice Modulation', analysis_results.get('voice_modulation', [])),
            ('Fluency and Intonation', analysis_results.get('fluency_and_intonation', [])),
            ('Clarity and Articulation', analysis_results.get('clarity_and_articulation', [])),
            ('Emotional Expression', analysis_results.get('emotional_expression', {})),
            ('Phrasing and Language', analysis_results.get('phrasing_and_language', {}))
        ]
        
        for section, data in sections:
            prompt += f"{section}: {json.dumps(data, indent=2)}\n\n"
        
        prompt += "Based on this analysis, provide 5 specific and actionable tips to improve the interviewee's skills. For each tip, explain why it's important and how it relates to the analysis. Format the response as a numbered list."
        
        return prompt

    def generate_tips(self, prompt: str, max_length: int, temperature: float, top_p: float) -> str:
        logger.info("Generating tips")
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Error generating tips: {str(e)}")
            raise

def main(args):
    try:
        # Initialize the tip generator
        tip_generator = InterviewTipGenerator(args.model_name, args.device)
        
        # Load the JSON file
        analysis_results = tip_generator.load_json_file(args.input_file)
        
        # Generate the prompt
        prompt = tip_generator.generate_prompt(analysis_results)
        
        # Generate tips using the LLM
        tips = tip_generator.generate_tips(prompt, args.max_length, args.temperature, args.top_p)
        
        # Print the generated tips
        logger.info("Generated Interview Improvement Tips:")
        print(tips)
        
        # Save the tips to a file
        output_file = args.output_file or "interview_improvement_tips.txt"
        with open(output_file, "w") as f:
            f.write(tips)
        logger.info(f"Tips saved to: {output_file}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate interview improvement tips using LLM")
    parser.add_argument("--input_file", required=True, help="Path to the JSON file containing interview analysis results")
    parser.add_argument("--output_file", help="Path to save the generated tips (default: interview_improvement_tips.txt)")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Name of the Hugging Face model to use")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"], help="Device to run the model on")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for text generation")
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P, help="Top-p value for text generation")
    
    args = parser.parse_args()
    main(args)