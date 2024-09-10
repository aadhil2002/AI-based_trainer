import json
import os
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
CONFIG = {
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "device": "auto",
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
    "input_file": r"D:\AI-based_trainer\data\interview_analysis_results.json",
    "output_file": r"interview_improvement_tips.txt",
    "api_key": os.getenv('GROQ_API_KEY')  # API key is now loaded from the .env file
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InterviewTipGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config['model_name']
        self.device = config['device']
        self.tokenizer = None
        self.model = None
        self.groq_client = None
        self.load_groq_client()

    def load_groq_client(self):
        logger.info(f"Loading Groq client for model: llama3-8b-8192")
        try:
            api_key = self.config.get('api_key') or os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("Groq API key not provided.")
            self.groq_client = Groq(api_key=api_key)
        except Exception as e:
            logger.error(f"Error initializing Groq client: {str(e)}")
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

    def generate_tips(self, prompt: str) -> str:
        logger.info("Generating tips")
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config['max_length'],
                    num_return_sequences=1,
                    temperature=self.config['temperature'],
                    top_p=self.config['top_p'],
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id  # Ensure pad token is used
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Error generating tips: {str(e)}")
            raise

    def groq_inference(self, prompt: str):
        logger.info("Running inference using Groq")
        completion = self.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config['temperature'],
            max_tokens=8192,
            top_p=self.config['top_p'],
            stream=True,
            stop=None,
        )
        
        result = ""
        for chunk in completion:
            result += chunk.choices[0].delta.content or ""
        
        return result

def main():
    try:
        # Initialize the tip generator
        tip_generator = InterviewTipGenerator(CONFIG)
        
        # Load the interview analysis results
        analysis_results = tip_generator.load_json_file(CONFIG['input_file'])
        
        # Generate the prompt
        prompt = tip_generator.generate_prompt(analysis_results)
        
        # Generate tips using the LLM (local or Groq)
        tips = tip_generator.groq_inference(prompt)
        
        # Print the generated tips
        logger.info("Generated Interview Improvement Tips:")
        print(tips)
        
        # Save the tips to a file
        output_file = CONFIG['output_file']
        with open(output_file, "w") as f:
            f.write(tips)
        logger.info(f"Tips saved to: {output_file}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
