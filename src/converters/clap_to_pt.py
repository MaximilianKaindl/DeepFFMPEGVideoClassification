#!/usr/bin/env python3
import torch
import os
import torchaudio
import torchaudio.transforms as T
import argparse
from typing import List, Tuple


class CLAPTraceWrapper(torch.nn.Module):
    """Wrapper class for CLAP model to make it traceable."""
    
    def __init__(self, clap_model, token_keys):
        super().__init__()
        self.clap = clap_model
        self.keys = token_keys
        
    def forward(self, audio, token1, token2=None):
        """Forward pass that handles the expected inputs."""
        with torch.no_grad():
            if token2 is not None:
                text_input = {self.keys[0]: token1, self.keys[1]: token2}
            else:
                text_input = {self.keys[0]: token1}
            
            return self.clap(audio, text_input)


def process_audio(audio_path: str, sample_rate: int, duration: float) -> torch.Tensor:
    """Process an audio file into a tensor suitable for the CLAP model."""
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


def process_text(model, labels: List[str], use_cuda: bool) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Process text labels into token tensors for the CLAP model."""
    # Get processed tokens
    tokens = model.preprocess_text(labels)
    
    # Move to GPU if needed
    if use_cuda and torch.cuda.is_available():
        tokens = {k: v.cuda() for k, v in tokens.items()}
    
    # Extract token keys and tensors
    token_keys = list(tokens.keys())
    
    # Get the input tensors based on available keys
    input_ids = tokens.get("input_ids", tokens.get(token_keys[0]))
    attention_mask = tokens.get("attention_mask", tokens.get(token_keys[1]) if len(token_keys) > 1 else None)
    
    return token_keys, input_ids.detach(), attention_mask.detach() if attention_mask is not None else None


def trace_and_save_model(wrapper: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output_path: str) -> None:
    """Trace the model and save it to the specified path."""
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, inputs, check_trace=False)
    
    traced_model.save(output_path)
    print(f"Model successfully traced and saved to {output_path}")
    print(f"Model file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


def export_clap_model(output_dir: str, version: str, use_cuda: bool, audio_path: str = 'resources/audio/blues.mp3', tokenizer_dir: str = None):
    """Main function to trace a CLAP model and save it."""
    try:
        from msclap import CLAP
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create tokenizer directory if provided and doesn't exist
        if tokenizer_dir:
            os.makedirs(tokenizer_dir, exist_ok=True)
        
        print(f"Loading CLAP model version {version}...")
        
        # Load the CLAP model
        model = CLAP(version=version, use_cuda=use_cuda)
        
        # Get model parameters
        sample_rate = getattr(model.args, 'sampling_rate', 44100)
        duration = getattr(model.args, 'duration', 7.0)
        
        print(f"Using sample rate: {sample_rate}Hz, duration: {duration}s")
        
        # Define test labels for tracing
        test_labels = ["rock", "jazz", "classical", "pop", "blues"]
        
        print(f"Processing audio file: {audio_path}")
        
        # Prepare inputs
        audio_sample = process_audio(audio_path, sample_rate, duration)
        if use_cuda and torch.cuda.is_available():
            audio_sample = audio_sample.cuda()
            print("Using CUDA for processing")
        else:
            print("Using CPU for processing")
        
        print("Processing text labels...")
        
        # Process text and get token tensors
        token_keys, input_ids, attention_mask = process_text(model, test_labels, use_cuda)
        
        # Save tokenizer to the specified directory or model directory
        tokenizer_save_dir = tokenizer_dir if tokenizer_dir else output_dir
        print(f"Saving tokenizer to {tokenizer_save_dir}")
        
        # Save tokenizer for future use
        model.tokenizer.save_pretrained(tokenizer_save_dir)
        
        # Save additional tokenizer metadata if we have a dedicated tokenizer directory
        if tokenizer_dir:
            import json
            
            # Create additional metadata file
            metadata = {
                "clap_version": version,
                "tokenizer_type": model.tokenizer.__class__.__name__,
                "vocab_size": model.tokenizer.vocab_size,
                "model_config": {
                    "sample_rate": sample_rate,
                    "duration": duration
                },
                "device_traced": "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
            }
            
            # Save metadata
            with open(os.path.join(tokenizer_dir, "clap_tokenizer_info.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Create wrapper and disable gradients
        wrapper = CLAPTraceWrapper(model.clap, token_keys)
        for param in wrapper.parameters():
            param.requires_grad = False
        
        # Define model output path
        model_path = os.path.join(output_dir, f"msclap{version}.pt")
        
        # Prepare inputs based on whether we have attention mask
        inputs = (audio_sample, input_ids, attention_mask) if attention_mask is not None else (audio_sample, input_ids)
        
        print(f"Tracing CLAP model...")
        
        # Trace and save the model
        trace_and_save_model(wrapper, inputs, model_path)
        
        print("\nModel Information:")
        print(f"- CLAP version: {version}")
        print(f"- Output directory: {output_dir}")
        print(f"- Model file: {model_path}")
        print(f"- Tokenizer saved to: {tokenizer_save_dir}")
        print(f"- Device used for tracing: {'CUDA' if use_cuda and torch.cuda.is_available() else 'CPU'}")
        
        return True
        
    except Exception as e:
        print(f"Error during export: {str(e)}")
        return False


def main():
    """Parse command line arguments and run the tracing process."""
    parser = argparse.ArgumentParser(description="Export CLAP model to TorchScript format")
    parser.add_argument("--version", type=str, default="2023", 
                        choices=["2022", "2023", "clapcap"],
                        help="CLAP model version to use")
    parser.add_argument("--output_dir", type=str, default="models/clap",
                        help="Directory to save traced models")
    parser.add_argument("--tokenizer_dir", type=str, default="models/clap/tokenizer_clap",
                        help="Directory to save tokenizer files")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA for tracing if available")
    parser.add_argument("--audio_path", type=str, default="resources/audio/blues.mp3",
                        help="Path to audio file for tracing")
    
    args = parser.parse_args()
    
    success = export_clap_model(args.output_dir, args.version, args.use_cuda, args.audio_path, args.tokenizer_dir)
    
    if success:
        print("\nExport completed successfully.")
        print("To test this model, use the test_scripted_models.py script.")
    else:
        print("\nExport failed.")


if __name__ == "__main__":
    main()