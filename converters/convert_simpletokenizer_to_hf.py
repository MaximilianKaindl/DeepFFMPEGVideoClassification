#!/usr/bin/env python3
import os
import json
import gzip
import torch
from typing import Dict, List, Optional, Union

class CLIPTokenizerExporter:
    """
    Helper class to export CLIP's SimpleTokenizer to a tokenizer.json format 
    compatible with Hugging Face tokenizers.
    """
    
    def __init__(self, simple_tokenizer):
        """
        Initialize with a SimpleTokenizer instance
        """
        self.tokenizer = simple_tokenizer
        
    def _extract_vocabulary(self) -> Dict[str, int]:
        """
        Extract vocabulary from the SimpleTokenizer
        """
        return self.tokenizer.encoder
    
    def _extract_merges(self) -> List[str]:
        """
        Extract BPE merges from the SimpleTokenizer's bpe_ranks
        """
        # Sort merges by their ranks
        sorted_merges = sorted(
            self.tokenizer.bpe_ranks.items(), 
            key=lambda item: item[1]
        )
        # Convert to strings in the format expected by tokenizer.json
        return [f"{first} {second}" for (first, second), _ in sorted_merges]
    
    def _extract_special_tokens(self) -> Dict[str, str]:
        """
        Extract special tokens mapping
        """
        special_tokens = {}
        for token_id in self.tokenizer.all_special_ids:
            token = self.tokenizer.decoder[token_id]
            if token == "<start_of_text>":
                special_tokens["bos_token"] = token
            elif token == "<end_of_text>":
                special_tokens["eos_token"] = token
        
        return special_tokens
    
    def create_tokenizer_json(self) -> Dict:
        """
        Create a tokenizer.json structure compatible with HF tokenizers
        """
        vocab = self._extract_vocabulary()
        merges = self._extract_merges()
        special_tokens = self._extract_special_tokens()
        
        # Create the tokenizer.json structure
        tokenizer_json = {
            "version": "1.0",
            "truncation": {
                "max_length": self.tokenizer.context_length,
                "strategy": "longest_first",
                "direction": "right",
                "stride": 0
            },
            "padding": {
                "strategy": "max_length",
                "direction": "right",
                "pad_to_multiple_of": 1,
                "pad_id": 0,
                "pad_type_id": 0,
                "pad_token": ""
            },
            "added_tokens": [
                {
                    "id": self.tokenizer.sot_token_id,
                    "special": True,
                    "content": "<start_of_text>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False
                },
                {
                    "id": self.tokenizer.eot_token_id,
                    "special": True,
                    "content": "<end_of_text>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False
                }
            ],
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    {
                        "type": "NFC"
                    },
                    {
                        "type": "Replace",
                        "pattern": " ",
                        "content": " "
                    }
                ]
            },
            "pre_tokenizer": {
                "type": "ByteLevel"
            },
            "post_processor": {
                "type": "ByteLevel"
            },
            "decoder": {
                "type": "ByteLevel"
            },
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": merges,
                "dropout": None,
                "fuse_unk": False,
                "byte_fallback": False
            }
        }
        
        return tokenizer_json
    
    def save_tokenizer(self, output_dir: str) -> None:
        """
        Save the tokenizer to the specified directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer.json
        tokenizer_json = self.create_tokenizer_json()
        with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
        
        # Save vocabulary.txt (as a reference)
        vocab = self._extract_vocabulary()
        with open(os.path.join(output_dir, "vocab.txt"), "w", encoding="utf-8") as f:
            for token, index in sorted(vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\n")
        
        # Save special_tokens_map.json
        special_tokens = self._extract_special_tokens()
        with open(os.path.join(output_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
            json.dump(special_tokens, f, ensure_ascii=False, indent=2)
        
        # Save tokenizer_config.json
        tokenizer_config = {
            "model_type": "clip",
            "tokenizer_class": "CLIPTokenizer",
            "bos_token": "<start_of_text>",
            "eos_token": "<end_of_text>",
            "context_length": self.tokenizer.context_length
        }
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        
        # Also save the BPE vocabulary file
        if hasattr(self.tokenizer, "bpe_path") and os.path.exists(self.tokenizer.bpe_path):
            try:
                # Copy the BPE vocabulary file
                with gzip.open(self.tokenizer.bpe_path, 'rb') as f_in:
                    with open(os.path.join(output_dir, "bpe_simple_vocab_16e6.txt"), 'wb') as f_out:
                        f_out.write(f_in.read())
                print(f"BPE vocabulary file saved to {output_dir}/bpe_simple_vocab_16e6.txt")
            except Exception as e:
                print(f"Could not save BPE vocabulary file: {e}")
        
        print(f"Tokenizer successfully saved to {output_dir}")


def patch_open_clip_export():
    """
    Patch the open_clip tokenizer to add save_pretrained method
    """
    import open_clip
    from functools import partial
    
    original_get_tokenizer = open_clip.get_tokenizer
    
    def patched_get_tokenizer(tokenizer_name="", context_length=77, *args, **kwargs):
        tokenizer = original_get_tokenizer(tokenizer_name, context_length, *args, **kwargs)
        
        # Add save_pretrained method to SimpleTokenizer
        if hasattr(tokenizer, 'tokenizer'):
            # This is a wrapper tokenizer with a .tokenizer attribute
            pass
        else:
            # This is a SimpleTokenizer
            tokenizer.save_pretrained = partial(save_simple_tokenizer, tokenizer)
            # Store the BPE path for reference
            from open_clip.tokenizer import default_bpe
            tokenizer.bpe_path = default_bpe()
        
        return tokenizer
    
    # Replace the original function
    open_clip.get_tokenizer = patched_get_tokenizer
    return patched_get_tokenizer


def save_simple_tokenizer(tokenizer, save_directory):
    """
    Save a SimpleTokenizer to the given directory
    """
    os.makedirs(save_directory, exist_ok=True)
    exporter = CLIPTokenizerExporter(tokenizer)
    exporter.save_tokenizer(save_directory)
    return save_directory


if __name__ == "__main__":
    import argparse
    import open_clip
    
    parser = argparse.ArgumentParser(description="Export CLIP SimpleTokenizer to tokenizer.json")
    parser.add_argument("--model_name", type=str, default="ViT-B-32", 
                        help="Name of the CLIP model to get tokenizer for")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save tokenizer files")
    args = parser.parse_args()
    
    # Patch the open_clip library to add save_pretrained method
    patch_open_clip_export()
    
    # Get the tokenizer
    tokenizer = open_clip.get_tokenizer(args.model_name)
    
    # Save the tokenizer
    tokenizer.save_pretrained(args.output_dir)
    print(f"SimpleTokenizer for {args.model_name} saved to {args.output_dir}")