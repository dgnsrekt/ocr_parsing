#!/usr/bin/env python3
# /// script
# dependencies = [
#   "requests>=2.31.0",
#   "Pillow>=10.0.0",
# ]
# requires-python = ">=3.8"
# ///
"""
DeepSeek-OCR Client for LM Studio
Comprehensive OCR experimentation script with multiple modes
"""

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
from PIL import Image

# DeepSeek-OCR prompt templates for LM Studio OpenAI-compatible API
# IMPORTANT NOTES:
# - For LM Studio's OpenAI API: Use clean prompts WITHOUT <image> or <|grounding|> tokens
# - Special tokens are only for native transformers API, not OpenAI format
# - LM Studio handles images through API structure, not text tokens
# - Recommended settings: temperature=0.0, max_tokens=8192
DEEPSEEK_PROMPTS = {
    "basic": "Extract all text from this image.",
    "detailed": "Extract all text from this image, preserving the layout and structure as much as possible.",
    "markdown": "Convert this document image to markdown format, preserving headings, lists, and formatting."
}


class DeepSeekOCRClient:
    """Client for DeepSeek-OCR model running on LM Studio."""

    def __init__(self, base_url: str = "http://llm1-studio.lan:1234/v1", debug: bool = False):
        """
        Initialize the OCR client.

        Args:
            base_url: LM Studio API base URL
            debug: Enable debug output
        """
        self.base_url = base_url.rstrip('/')
        self.chat_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"
        self.debug = debug

    def preprocess_image(
        self,
        image_path: str,
        max_size: int = 2048,
        output_path: Optional[str] = None
    ) -> str:
        """
        Preprocess and optionally resize image for better performance.

        Args:
            image_path: Path to the image file
            max_size: Maximum dimension (width or height)
            output_path: Optional path to save resized image

        Returns:
            Path to the processed image
        """
        try:
            img = Image.open(image_path)
            original_size = img.size

            # Check if resizing is needed
            if max(img.size) > max_size:
                # Calculate new size maintaining aspect ratio
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)

                img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Save to temp file or specified output
                if output_path is None:
                    path = Path(image_path)
                    output_path = str(path.parent / f"{path.stem}_resized{path.suffix}")

                img.save(output_path, quality=95, optimize=True)

                print(f"  Resized: {original_size} → {new_size}")
                return output_path
            else:
                print(f"  Size OK: {original_size} (no resize needed)")
                return image_path

        except Exception as e:
            print(f"  Warning: Preprocessing failed: {e}")
            return image_path

    def encode_image(self, image_path: str) -> str:
        """
        Encode image file to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string with data URI prefix
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')

        # Detect image type from extension
        ext = path.suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')

        return f"data:{mime_type};base64,{encoded}"

    def test_connection(self) -> bool:
        """
        Test connection to LM Studio server and check for deepseek-ocr model.

        Returns:
            True if connection successful and model available
        """
        print(f"Testing connection to: {self.base_url}")
        print("-" * 70)

        # Test server reachability
        try:
            response = requests.get(self.models_url, timeout=5)
            print(f"✓ Server is reachable (Status: {response.status_code})")
        except requests.ConnectionError:
            print("✗ Cannot connect to server")
            print("  Troubleshooting:")
            print("    - Check if LM Studio is running")
            print("    - Verify the hostname: llm1-studio.lan")
            print("    - Ensure port 1234 is accessible")
            return False
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False

        # List available models
        try:
            models_data = response.json()

            if self.debug:
                print(f"\n[DEBUG] Models endpoint response:")
                print(json.dumps(models_data, indent=2))

            models = [m['id'] for m in models_data.get('data', [])]
            print(f"✓ Available models: {len(models)}")
            for model in models:
                print(f"    - {model}")

            # Check for deepseek-ocr and find exact name
            deepseek_models = [m for m in models if 'deepseek' in m.lower() and 'ocr' in m.lower()]
            if deepseek_models:
                print(f"✓ DeepSeek-OCR model found!")
                print(f"  Exact model name: {deepseek_models[0]}")
                if len(deepseek_models) > 1:
                    print(f"  Note: Multiple DeepSeek-OCR models available: {deepseek_models}")
            else:
                print("⚠ DeepSeek-OCR model not found")
                print("  Make sure deepseek-ocr is loaded in LM Studio")
                if models:
                    print(f"  Available models are: {models}")
                return False

        except Exception as e:
            print(f"✗ Failed to parse models response: {e}")
            if self.debug:
                print(f"[DEBUG] Response text: {response.text[:500]}")
            return False

        # Test chat endpoint
        try:
            test_payload = {
                "model": models[0] if models else "deepseek-ocr",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }

            if self.debug:
                print(f"\n[DEBUG] Testing chat endpoint with payload:")
                print(json.dumps(test_payload, indent=2))

            response = requests.post(
                self.chat_url,
                json=test_payload,
                timeout=30
            )

            if self.debug:
                print(f"[DEBUG] Chat response status: {response.status_code}")
                try:
                    print(f"[DEBUG] Chat response:")
                    print(json.dumps(response.json(), indent=2))
                except:
                    print(f"[DEBUG] Chat response text: {response.text[:500]}")

            if response.status_code == 200:
                print("✓ Chat completions endpoint working")
            else:
                print(f"⚠ Chat endpoint returned status {response.status_code}")
                if self.debug:
                    print(f"[DEBUG] Response: {response.text[:500]}")
        except Exception as e:
            print(f"✗ Chat completions test failed: {e}")
            if self.debug and 'response' in locals():
                print(f"[DEBUG] Response text: {response.text[:500]}")
            return False

        print("-" * 70)
        print("✓ All tests passed! Ready to process images.")
        return True

    def _create_vision_payload(
        self,
        image_data: str,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        format_type: str = "openai_standard"
    ) -> Dict[str, Any]:
        """
        Create API payload in different formats for compatibility.

        Args:
            image_data: Base64 encoded image (with or without data URI prefix)
            prompt: Text prompt
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens in response
            format_type: Which format to use

        Returns:
            API payload dictionary
        """
        # Extract just the base64 data without prefix if present
        base64_only = image_data.split(',', 1)[1] if ',' in image_data else image_data

        if format_type == "openai_standard":
            # Standard OpenAI Vision API format
            return {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data}}
                        ]
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

        elif format_type == "base64_only":
            # Try with just base64, no data URI prefix
            return {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": base64_only}}
                        ]
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

        elif format_type == "simple_content":
            # Simplified content structure (text + image concatenated)
            return {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nImage: {base64_only[:100]}..."
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

        elif format_type == "image_field":
            # Direct image field (some APIs use this)
            return {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "image": base64_only
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

        elif format_type == "llama_cpp_vision":
            # LLaMA.cpp style vision format (used by some LM Studio models)
            return {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "data": base64_only}
                        ]
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

        else:
            raise ValueError(f"Unknown format_type: {format_type}")

    def extract_text(
        self,
        image_path: str,
        model: str = "deepseek-ocr",
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        preprocess: bool = True,
        max_image_size: int = 2048,
        api_format: str = "auto"
    ) -> str:
        """
        Extract text from an image using DeepSeek-OCR.

        Args:
            image_path: Path to the image file
            model: Model name (default: "deepseek-ocr")
            prompt: Custom prompt
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
            preprocess: Whether to preprocess/resize image
            max_image_size: Max dimension for preprocessing
            api_format: API format to use ("auto", "openai_standard", "llama_cpp_vision", etc.)

        Returns:
            Extracted text from the image
        """
        if prompt is None:
            prompt = DEEPSEEK_PROMPTS["detailed"]

        # Preprocess image if requested
        processed_path = image_path
        if preprocess:
            processed_path = self.preprocess_image(image_path, max_size=max_image_size)

        # Encode image to base64
        image_data = self.encode_image(processed_path)

        # Determine which formats to try
        if api_format == "auto":
            # Try formats in order of likelihood
            formats_to_try = [
                "llama_cpp_vision",  # Most likely for LM Studio
                "openai_standard",   # Standard OpenAI format
                "image_field",       # Direct image field
                "base64_only",       # Base64 without data URI
            ]
        else:
            formats_to_try = [api_format]

        last_error = None
        extracted_text = ""

        for fmt in formats_to_try:
            if self.debug:
                print(f"\n[DEBUG] Trying API format: {fmt}")

            try:
                # Create payload for this format
                payload = self._create_vision_payload(
                    image_data, prompt, model, temperature, max_tokens, fmt
                )

                # Debug: Show request details
                if self.debug:
                    print(f"[DEBUG] API Request:")
                    print(f"  URL: {self.chat_url}")
                    print(f"  Model: {model}")
                    print(f"  Format: {fmt}")
                    print(f"  Prompt: {prompt[:100]}...")
                    print(f"  Temperature: {temperature}")
                    print(f"  Max tokens: {max_tokens}")
                    print(f"  Image size: ~{len(image_data)} chars (base64)")

                # Send request to LM Studio
                response = requests.post(
                    self.chat_url,
                    json=payload,
                    timeout=120  # OCR can take a while
                )

                # Debug: Show response status
                if self.debug:
                    print(f"[DEBUG] Response Status: {response.status_code}")

                response.raise_for_status()

                # Parse response
                result = response.json()

                # Debug: Show full response
                if self.debug:
                    print(f"[DEBUG] Full API Response:")
                    print(json.dumps(result, indent=2))

                extracted_text = result['choices'][0]['message']['content']

                # Debug: Show extracted text info
                if self.debug:
                    print(f"[DEBUG] Extracted text length: {len(extracted_text)} characters")

                # If we got text, success!
                if extracted_text.strip():
                    if self.debug:
                        print(f"[DEBUG] ✓ Format '{fmt}' worked! Got {len(extracted_text)} characters")

                    # Clean up temporary resized image if created
                    if preprocess and processed_path != image_path:
                        Path(processed_path).unlink(missing_ok=True)

                    return extracted_text.strip()
                else:
                    if self.debug:
                        print(f"[DEBUG] ✗ Format '{fmt}' returned empty content")
                    # Try next format
                    continue

            except requests.RequestException as e:
                last_error = e
                if self.debug:
                    print(f"[DEBUG] ✗ Format '{fmt}' failed with request error: {e}")
                    if 'response' in locals():
                        print(f"[DEBUG] Response text: {response.text[:500]}")
                # Try next format
                continue

            except (KeyError, IndexError) as e:
                last_error = e
                if self.debug:
                    print(f"[DEBUG] ✗ Format '{fmt}' failed with parse error: {e}")
                    if 'result' in locals():
                        print(f"[DEBUG] Response structure: {json.dumps(result, indent=2)[:500]}")
                # Try next format
                continue

        # If we get here, all formats failed
        if self.debug:
            print(f"\n[DEBUG] All API formats failed to extract text!")

        # Clean up temporary resized image if created
        if preprocess and processed_path != image_path:
            Path(processed_path).unlink(missing_ok=True)

        if last_error:
            raise last_error
        else:
            # All formats returned empty
            return ""


def save_output(content: str, output_path: str, format_type: str = "txt"):
    """
    Save extracted text to a file.

    Args:
        content: Text content to save
        output_path: Path to save file
        format_type: Output format (txt, json, md)
    """
    path = Path(output_path)

    if format_type == "json":
        data = {"text": content, "length": len(content)}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif format_type == "md":
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# OCR Result\n\n")
            f.write(content)
    else:  # txt
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"  Saved to: {path}")


def mode_single_image(args):
    """Process a single image."""
    client = DeepSeekOCRClient(base_url=args.base_url, debug=args.debug)

    print(f"\nProcessing: {args.image}")
    print("=" * 70)

    try:
        text = client.extract_text(
            args.image,
            model=args.model,
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            preprocess=not args.no_preprocess,
            max_image_size=args.max_size,
            api_format=args.api_format
        )

        print("\nExtracted Text:")
        print("-" * 70)
        print(text)
        print("-" * 70)

        # Save output if requested
        if args.output:
            save_output(text, args.output, args.format)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def mode_batch_processing(args):
    """Process multiple images in a directory."""
    client = DeepSeekOCRClient(base_url=args.base_url, debug=args.debug)

    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
    batch_dir = Path(args.batch)

    if not batch_dir.exists():
        print(f"Error: Directory not found: {args.batch}", file=sys.stderr)
        sys.exit(1)

    # Find all images
    images = [f for f in batch_dir.iterdir()
              if f.suffix.lower() in image_extensions]

    if not images:
        print(f"No images found in: {args.batch}")
        return

    print(f"\nFound {len(images)} images to process")
    print("=" * 70)

    results = {}

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {image_path.name}")

        try:
            text = client.extract_text(
                str(image_path),
                model=args.model,
                prompt=args.prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                preprocess=not args.no_preprocess,
                max_image_size=args.max_size,
                api_format=args.api_format
            )

            results[image_path.name] = {
                'success': True,
                'text': text,
                'length': len(text)
            }

            print(f"  ✓ Extracted {len(text)} characters")

        except Exception as e:
            results[image_path.name] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ✗ Failed: {e}")

    # Save batch results
    output_file = args.output or "batch_ocr_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Batch processing complete!")
    print(f"Results saved to: {output_file}")

    # Summary
    success_count = sum(1 for r in results.values() if r['success'])
    print(f"Successful: {success_count}/{len(images)}")


def mode_structured_extraction(args):
    """Extract structured data from an image."""
    client = DeepSeekOCRClient(base_url=args.base_url, debug=args.debug)

    # Custom prompt for structured extraction
    structured_prompt = (
        "Extract all text from this image and structure it as JSON with the following fields:\n"
        "- title: The main heading or title if present\n"
        "- date: Any dates found (keep original format)\n"
        "- amounts: List of any monetary amounts or numbers\n"
        "- items: List of items, products, or line items if present\n"
        "- text_content: All other relevant text organized by sections\n"
        "- metadata: Any other structured information (addresses, phone numbers, etc.)\n\n"
        "Return ONLY valid JSON. If a field is not applicable, use null or an empty array."
    )

    prompt = args.prompt or structured_prompt

    print(f"\nProcessing for structured extraction: {args.structured}")
    print("=" * 70)

    try:
        text = client.extract_text(
            args.structured,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            preprocess=not args.no_preprocess,
            max_image_size=args.max_size,
            api_format=args.api_format
        )

        # Try to parse as JSON
        try:
            structured_data = json.loads(text)
            print("\nStructured Data:")
            print("-" * 70)
            print(json.dumps(structured_data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("\nExtracted Text (not valid JSON):")
            print("-" * 70)
            print(text)
            structured_data = {'raw_text': text}

        # Save output if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR Client for LM Studio - Image to Text Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Test connection:
    %(prog)s --test

  Process single image:
    %(prog)s --image photo.jpg
    %(prog)s --image invoice.png --output result.txt

  Batch process directory:
    %(prog)s --batch ./images --output results.json

  Structured extraction:
    %(prog)s --structured receipt.jpg --output data.json

  Custom prompt:
    %(prog)s --image doc.png --prompt "Extract only email addresses"
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--test', action='store_true',
                           help='Test connection to LM Studio server')
    mode_group.add_argument('--image', type=str, metavar='PATH',
                           help='Process a single image')
    mode_group.add_argument('--batch', type=str, metavar='DIR',
                           help='Process all images in a directory')
    mode_group.add_argument('--structured', type=str, metavar='PATH',
                           help='Extract structured data (JSON) from image')

    # Common options
    parser.add_argument('--base-url', type=str,
                       default='http://llm1-studio.lan:1234/v1',
                       help='LM Studio API base URL (default: http://llm1-studio.lan:1234/v1)')
    parser.add_argument('--model', type=str, default='deepseek-ocr',
                       help='Model name (default: deepseek-ocr)')
    parser.add_argument('--prompt', type=str,
                       help='Custom extraction prompt')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=8192,
                       help='Maximum tokens in response (default: 8192, recommended for DeepSeek-OCR)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path')
    parser.add_argument('--format', type=str,
                       choices=['txt', 'json', 'md'],
                       default='txt',
                       help='Output format for single image mode (default: txt)')

    # Image preprocessing
    parser.add_argument('--no-preprocess', action='store_true',
                       help='Disable automatic image preprocessing/resizing')
    parser.add_argument('--max-size', type=int, default=2048,
                       help='Maximum image dimension for preprocessing (default: 2048)')

    # Debug and API options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output (show API requests/responses)')
    parser.add_argument('--api-format', type=str, default='auto',
                       choices=['auto', 'openai_standard', 'llama_cpp_vision', 'image_field', 'base64_only'],
                       help='API format to use (default: auto - tries all formats)')

    args = parser.parse_args()

    # Execute appropriate mode
    if args.test:
        client = DeepSeekOCRClient(base_url=args.base_url, debug=args.debug)
        success = client.test_connection()
        sys.exit(0 if success else 1)
    elif args.image:
        mode_single_image(args)
    elif args.batch:
        mode_batch_processing(args)
    elif args.structured:
        mode_structured_extraction(args)


if __name__ == "__main__":
    main()
