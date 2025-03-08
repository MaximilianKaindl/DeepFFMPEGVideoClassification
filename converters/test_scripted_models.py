#!/usr/bin/env python3
import argparse
import os
import torch
import torchaudio
import torchaudio.transforms as T
from PIL import Image
import sys
import json

def load_model(model_path):
    """Load a scripted model from the specified path."""
    try:
        model = torch.jit.load(model_path)
        model.eval()
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def test_clip_model(model, preprocess, tokenizer, image_path, candidate_texts):
    """Test a scripted CLIP model with an image and candidate texts."""
    try:
        # Process image
        image = Image.open(image_path)
        image_input = preprocess(image).unsqueeze(0)
        
        # Process text
        text_tokens = tokenizer(candidate_texts)
        
        with torch.no_grad():
            # Encode image and normalize
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Encode text and normalize
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Create results dictionary
        results = {text: float(score) for text, score in zip(candidate_texts, similarity[0].tolist())}
        
        # Print results sorted by confidence
        print("\nCLIP Classification Results:")
        for text, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{text}: {score:.2%} confidence")
            
        return results
        
    except Exception as e:
        print(f"Error testing CLIP model: {e}")
        return None

def process_audio(audio_path, sample_rate=44100, duration=7.0):
    """Process an audio file for CLAP model input."""
    try:
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != sample_rate:
            waveform = T.Resample(sr, sample_rate)(waveform)
        
        # Convert stereo to mono if needed
        audio = waveform.mean(dim=0) if waveform.size(0) > 1 else waveform.squeeze(0)
        
        # Adjust length to match target duration
        target_length = int(sample_rate * duration)
        if audio.size(0) < target_length:
            audio = torch.nn.functional.pad(audio, (0, target_length - audio.size(0)))
        elif audio.size(0) > target_length:
            audio = audio[:target_length]
        
        # Add batch dimension
        return audio.unsqueeze(0)
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def process_text_for_clap(tokenizer, labels, output_dir):
    """Process text labels for CLAP model input."""
    try:
        # Get tokenizer from saved directory
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Tokenize text
        tokens = tokenizer(labels, padding=True, truncation=True, return_tensors="pt")
        
        # Extract token tensors
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        
        return input_ids, attention_mask
    except Exception as e:
        print(f"Error processing text for CLAP: {e}")
        return None, None

def test_clap_model(model, audio_path, labels, sample_rate=44100, duration=7.0, use_cuda=False):
    """Test a scripted CLAP model with audio and text labels."""
    try:
        # Process audio
        audio_sample = process_audio(audio_path, sample_rate, duration)
        if audio_sample is None:
            return None
            
        if use_cuda and torch.cuda.is_available():
            audio_sample = audio_sample.cuda()
            model = model.cuda()
        
        # Get token info from model structure
        # In most CLAP cases, we need input_ids and might need attention_mask
        token_dict = {}
        for label in labels:
            token_dict[label] = {}
        
        # Run model inference
        with torch.no_grad():
            # We assume the model expects (audio, input_ids, attention_mask)
            # or just (audio, input_ids) depending on the model structure
            try:
                # Try different input patterns based on model requirements
                # This is challenging without knowing exactly how the model was traced
                
                # Option 1: Process all text at once
                # This might work with some traced models
                from msclap import CLAP
                clap = CLAP(use_cuda=use_cuda)
                processed_tokens = clap.preprocess_text(labels)
                if use_cuda and torch.cuda.is_available():
                    processed_tokens = {k: v.cuda() for k, v in processed_tokens.items()}
                
                outputs = model(audio_sample, processed_tokens)
                
                # Extract results
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    caption_embed, audio_embed, logit_scale = outputs[:3]
                else:
                    print("Unexpected output format from CLAP model")
                    return None
                
                # Normalize embeddings
                caption_embed = caption_embed / caption_embed.norm(dim=-1, keepdim=True)
                audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (logit_scale * (audio_embed @ caption_embed.T)).softmax(dim=-1)
                similarity_np = similarity.cpu().numpy()
                
                # Create results dictionary
                results = {label: float(similarity_np[0][i]) for i, label in enumerate(labels)}
                
                # Print results
                print("\nCLAP Audio Classification Results:")
                for label, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
                    print(f"{label}: {score:.4f} confidence")
                
                return results
                
            except Exception as e:
                print(f"Error during CLAP inference: {e}")
                print("This could be due to model structure mismatch. Try providing the exact same inputs used during tracing.")
                return None
                
    except Exception as e:
        print(f"Error testing CLAP model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test scripted CLIP or CLAP models")
    
    # General arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the scripted model file")
    parser.add_argument("--model_type", type=str, required=True, choices=["clip", "clap"],
                        help="Type of model to test (clip or clap)")
    parser.add_argument("--output_file", type=str,
                        help="Path to save test results as JSON")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")
                        
    # CLIP specific arguments
    parser.add_argument("--image_path", type=str,
                        help="Path to image file for CLIP testing")
    parser.add_argument("--clip_model_name", type=str, 
                        help="CLIP model name for tokenizer and preprocessing")
    parser.add_argument("--candidates", type=str, nargs="+",
                       default=["a photo of a dog", "a photo of a cat", 
                                "a photo of a car", "a photo of a house", 
                                "a photo of a person"],
                       help="Text descriptions for CLIP classification")
                       
    # CLAP specific arguments
    parser.add_argument("--audio_path", type=str,
                        help="Path to audio file for CLAP testing")
    parser.add_argument("--sample_rate", type=int, default=44100,
                        help="Sample rate for audio processing")
    parser.add_argument("--duration", type=float, default=7.0,
                        help="Duration for audio processing")
    parser.add_argument("--output_dir", type=str,
                        help="Directory with saved tokenizer for CLAP")
    parser.add_argument("--labels", type=str, nargs="+",
                       default=["rock", "jazz", "classical", "pop", "blues"],
                       help="Audio labels for CLAP classification")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Test based on model type
    results = None
    if args.model_type == "clip":
        if not args.image_path:
            print("Error: --image_path is required for CLIP testing")
            sys.exit(1)
        
        if not args.clip_model_name:
            print("Error: --clip_model_name is required for CLIP testing")
            sys.exit(1)
            
        # Import here to avoid dependency issues if not using CLIP
        import open_clip
        _, preprocess, _ = open_clip.create_model_and_transforms(args.clip_model_name)
        tokenizer = open_clip.get_tokenizer(args.clip_model_name)
        
        results = test_clip_model(model, preprocess, tokenizer, 
                                args.image_path, args.candidates)
                                
    elif args.model_type == "clap":
        if not args.audio_path:
            print("Error: --audio_path is required for CLAP testing")
            sys.exit(1)
        
        results = test_clap_model(model, args.audio_path, args.labels,
                                args.sample_rate, args.duration, args.use_cuda)
    
    # Save results if requested
    if results and args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()