# DeepSeek-OCR Client for LM Studio

A comprehensive Python script for extracting text from images using the DeepSeek-OCR model running on LM Studio.

## Features

- **Multiple Modes**: Single image, batch processing, and structured data extraction
- **Connection Testing**: Verify LM Studio server and model availability
- **Image Preprocessing**: Automatic resizing for better performance
- **Flexible Output**: Console, text files, JSON, or Markdown
- **Customizable Prompts**: Tailor extraction behavior to your needs
- **UV Compatible**: Modern Python dependency management

## Prerequisites

- Python 3.8 or higher
- [UV](https://github.com/astral-sh/uv) package manager
- LM Studio running with DeepSeek-OCR model loaded
- DeepSeek-OCR model accessible at `llm1-studio.lan:1234` (or configure custom URL)

## Installation

No installation needed! The script uses inline dependency declarations (PEP 723), so UV automatically manages dependencies:

```bash
# UV will auto-install dependencies on first run
uv run deepseek_ocr.py --test
```

Dependencies are declared directly in the script, so you don't need a separate `pyproject.toml` or `requirements.txt` file.

### Verify LM Studio Setup:

Make sure DeepSeek-OCR is loaded in LM Studio:

```bash
ssh llm1-studio.lan
lms ps
# Should show: deepseek-ocr model loaded
```

## Usage

### Test Connection

Verify that the LM Studio server is accessible and the model is loaded:

```bash
uv run deepseek_ocr.py --test
```

Expected output:
```
Testing connection to: http://llm1-studio.lan:1234/v1
----------------------------------------------------------------------
✓ Server is reachable (Status: 200)
✓ Available models: 1
    - deepseek-ocr
✓ DeepSeek-OCR model found!
✓ Chat completions endpoint working
----------------------------------------------------------------------
✓ All tests passed! Ready to process images.
```

### Process Single Image

Extract text from a single image:

```bash
# Basic usage
uv run deepseek_ocr.py --image photo.jpg

# Save to file
uv run deepseek_ocr.py --image invoice.png --output result.txt

# Save as JSON
uv run deepseek_ocr.py --image doc.jpg --output result.json --format json

# Save as Markdown
uv run deepseek_ocr.py --image notes.png --output result.md --format md
```

### Batch Processing

Process all images in a directory:

```bash
# Process all images in ./images directory
uv run deepseek_ocr.py --batch ./images

# Custom output file
uv run deepseek_ocr.py --batch ./documents --output ocr_results.json
```

Output will be a JSON file with results for each image:

```json
{
  "invoice_001.png": {
    "success": true,
    "text": "Extracted text here...",
    "length": 1234
  },
  "receipt_002.jpg": {
    "success": true,
    "text": "More text...",
    "length": 567
  }
}
```

### Structured Data Extraction

Extract structured information from images (invoices, receipts, forms):

```bash
# Extract as JSON
uv run deepseek_ocr.py --structured invoice.png --output data.json

# With custom prompt
uv run deepseek_ocr.py --structured receipt.jpg \
  --prompt "Extract vendor name, total amount, and date as JSON" \
  --output receipt_data.json
```

Example output:
```json
{
  "title": "Invoice #12345",
  "date": "2025-11-19",
  "amounts": ["$150.00", "$15.00", "$165.00"],
  "items": [
    "Web Development Services - $150.00",
    "Tax - $15.00"
  ],
  "text_content": {
    "vendor": "Acme Corp",
    "customer": "John Doe"
  },
  "metadata": {
    "invoice_number": "12345",
    "due_date": "2025-12-19"
  }
}
```

## Advanced Usage

### Custom Prompts

Tailor the extraction behavior:

```bash
# Extract only email addresses
uv run deepseek_ocr.py --image business_card.jpg \
  --prompt "Extract only email addresses from this image"

# Extract in a specific format
uv run deepseek_ocr.py --image menu.png \
  --prompt "Extract the menu items with prices in a numbered list format"

# Extract specific fields
uv run deepseek_ocr.py --image form.jpg \
  --prompt "Extract: Name, Address, Phone Number, Email"
```

### Image Preprocessing

Control automatic image preprocessing:

```bash
# Disable preprocessing (use original image)
uv run deepseek_ocr.py --image large_photo.jpg --no-preprocess

# Set custom maximum size (default: 2048px)
uv run deepseek_ocr.py --image document.png --max-size 1024
```

Preprocessing automatically resizes large images while maintaining aspect ratio, which can significantly improve processing speed.

### Custom LM Studio Server

If your LM Studio instance is at a different location:

```bash
uv run deepseek_ocr.py --image photo.jpg \
  --base-url http://localhost:1234/v1
```

### Adjust Model Parameters

Fine-tune the model behavior:

```bash
# Increase max tokens for long documents
uv run deepseek_ocr.py --image long_article.jpg --max-tokens 4000

# Use higher temperature for more creative extraction
uv run deepseek_ocr.py --image handwritten_note.jpg --temperature 0.3

# Specify exact model name
uv run deepseek_ocr.py --image doc.png --model deepseek-ocr-v2
```

## Supported Image Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- GIF (`.gif`)
- WebP (`.webp`)

## Troubleshooting

### Connection Issues

If you get connection errors:

1. **Check LM Studio is running**:
   ```bash
   ssh llm1-studio.lan
   lms status
   ```

2. **Verify model is loaded**:
   ```bash
   lms ps
   # Should show: deepseek-ocr model in IDLE or RUNNING state
   ```

3. **Test network connectivity**:
   ```bash
   ping llm1-studio.lan
   curl http://llm1-studio.lan:1234/v1/models
   ```

4. **Check firewall**:
   Make sure port 1234 is not blocked

### Model Not Found

If the test shows model not found:

```bash
# On llm1-studio.lan
lms load deepseek-ocr
```

### Timeout Errors

For large images or slow processing:

- Enable preprocessing: Remove `--no-preprocess` flag
- Reduce max-size: `--max-size 1024`
- Increase timeout in code if needed (default: 120s)

### Poor OCR Quality

To improve OCR results:

- Use higher resolution images
- Ensure good contrast and lighting
- Try different prompts
- Use `--no-preprocess` to keep original quality
- Experiment with temperature values

## API Format Reference

DeepSeek-OCR uses OpenAI-compatible vision API format:

### Request Format

```json
{
  "model": "deepseek-ocr",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Extract all text from this image"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
          }
        }
      ]
    }
  ],
  "temperature": 0.0,
  "max_tokens": 2000
}
```

### Response Format

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "deepseek-ocr",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Extracted text content here..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 50,
    "total_tokens": 200
  }
}
```

## Examples

### Extract Text from Screenshot

```bash
uv run deepseek_ocr.py --image screenshot.png --output screenshot_text.txt
```

### Process Scanned Documents

```bash
# Single document
uv run deepseek_ocr.py --image scanned_contract.pdf.jpg --output contract.txt

# Batch of scanned pages
uv run deepseek_ocr.py --batch ./scanned_pages --output scan_results.json
```

### Extract Data from Receipts

```bash
uv run deepseek_ocr.py --structured receipt.jpg \
  --prompt "Extract: merchant name, date, total amount, items purchased as JSON" \
  --output receipt.json
```

### Handwritten Notes

```bash
uv run deepseek_ocr.py --image handwritten.jpg \
  --prompt "Transcribe this handwritten text" \
  --temperature 0.2 \
  --output notes.txt
```

## Project Structure

```
ocr_parsing/
├── deepseek_ocr.py       # Main script with inline dependencies
└── README.md              # This file
```

The script uses PEP 723 inline script metadata for dependency declarations, so no separate configuration files are needed.

## Performance Tips

1. **Image Size**: Preprocessing to 2048px max dimension provides good balance of quality and speed
2. **Batch Processing**: Process multiple images in one session to amortize startup costs
3. **Network**: Use wired connection to LM Studio host for better performance
4. **Temperature**: Use 0.0 for deterministic, consistent results
5. **Max Tokens**: Set appropriately - too low truncates results, too high wastes time

## License

This is a utility script for personal/experimental use with DeepSeek-OCR and LM Studio.

## Resources

- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [LM Studio](https://lmstudio.ai/)
- [UV Package Manager](https://github.com/astral-sh/uv)

## Contributing

This is an experimental script. Feel free to modify and extend for your use cases!

## Changelog

### v0.1.0 (2025-11-19)
- Initial release
- Single image, batch, and structured extraction modes
- Connection testing
- Image preprocessing
- Multiple output formats
