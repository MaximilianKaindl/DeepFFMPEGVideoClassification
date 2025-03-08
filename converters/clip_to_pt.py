#!/usr/bin/env python3
import torch
import argparse
import os
import open_clip
import convert_simpletokenizer_to_hf as exporter_module

def export_clip_model(model_name, dataset_name, output_path, tokenizer_dir=None):
    """Export a CLIP model to TorchScript format."""
    try:
        print(f"Loading CLIP model {model_name} trained on {dataset_name}...")
        
        # Create model with pretrained weights
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=dataset_name
        )
        model.eval()
        # Get tokenizer
        tokenizer = open_clip.get_tokenizer(model_name)

        if tokenizer_dir:
            try:
                # Try to use save_pretrained directly
                if hasattr(tokenizer, 'save_pretrained'):
                    print(f"Saving tokenizer to {tokenizer_dir}...")
                    tokenizer.save_pretrained(tokenizer_dir)
                # If SimpleTokenizer without save_pretrained method
                                    
                else:
                    print(f"Using custom exporter to save tokenizer to {tokenizer_dir}...")
                    exporter_module.save_simple_tokenizer(tokenizer, tokenizer_dir)
            except Exception as e:
                print(f"Error saving tokenizer: {str(e)}")        

        # Script the model
        print("Converting model to TorchScript format...")
        scripted_model = torch.jit.script(model)
        
        # Save the model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scripted_model.save(output_path)
        print(f"Model successfully exported to {output_path}")
        
        # Print model information
        print(f"\nModel Information:")
        print(f"- Model: {model_name}")
        print(f"- Dataset: {dataset_name}")
        print(f"- Output file: {output_path}")
        print(f"- File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        if tokenizer_dir:
            print(f"- Tokenizer directory: {tokenizer_dir}")
        
        return True
    
    except Exception as e:
        print(f"Error during export: {str(e)}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Export CLIP model to TorchScript format')
    parser.add_argument('--model_name', type=str, required=True, 
                        help='Name of the CLIP model (e.g., ViT-B-32)')
    parser.add_argument('--dataset_name', type=str, required=True, 
                        help='Name of the training dataset (e.g., laion2b_s34b_b79k)')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Path for output TorchScript model')
    parser.add_argument('--tokenizer_dir', type=str, default="../tokenizer_clip",
                        help='Directory to save tokenizer files')
    parser.add_argument('--list_models', action='store_true',
                        help='List available models and exit')
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        print("Available CLIP models:")
        models_and_datasets = open_clip.list_pretrained()
        for model, dataset in models_and_datasets:
            print(f"- {model} (trained on {dataset})")
        return
    
    # Export the model
    success = export_clip_model(
        args.model_name, 
        args.dataset_name, 
        args.output_path,
        args.tokenizer_dir
    )
    
    if success:
        print("\nExport completed successfully.")
        print("To test this model, use the test_scripted_models.py script.")
    else:
        print("\nExport failed.")


if __name__ == "__main__":
    main()