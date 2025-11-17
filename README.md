# Build a Small Language Model from Scratch

## Overview

This project provides a complete workflow for building, training, and evaluating a small transformer-based language model (SLM) from scratch, targeting approximately 50-60 million parameters. The model is inspired by GPT architectures (nanoGPT/GPT-2) and is trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset—a collection of short synthetic stories suitable for young children.

## Features

- **Custom Transformer Model**: Build a compact transformer with configurable layers, heads, and embedding sizes.
- **Efficient Tokenization**: Uses OpenAI's `tiktoken` (GPT-2) for fast, accurate tokenization.
- **Training Data Pipelines**: Large training/validation sets are stored efficiently using binary formats (`.bin`), supporting out-of-RAM learning.
- **Modern Training Practices**:
    - AdamW optimizer with weight decay
    - Mixed precision training (`bfloat16`/`float16`)
    - Learning rate warmup and cosine decay scheduler
    - Gradient accumulation and clipping for stability
- **Evaluation and Visualization**: Track and plot train/validation loss curves.
- **Inference Demo**: Generate creative stories with the trained model.

## Workflow

1. **Dataset Preparation**:  
    - Downloads TinyStories from HuggingFace.
    - Tokenizes and stores dataset as `.bin` files to disk.

2. **Batch Creation**:  
    - Dynamically creates batches for context window training using efficient memory mapping.

3. **Model Architecture**:  
    - Implements a compact GPT-like transformer: self-attention, MLPs, layer norm, residual connections.

4. **Training**:  
    - Set up configs, optimizer, scheduler, and autograd/scaler.
    - Train the SLM while checkpointing the best model.

5. **Evaluation**:  
    - Regularly validates loss and visualizes results.

6. **Inference**:  
    - Load best model and generate text continuations from a prompt.

## Usage

> **Note**: The workflow is provided as a Jupyter Notebook.  
> Recommended: Run on Google Colab or any machine with a CUDA GPU for reasonable training times.

1. **Install Dependencies**  
    - Python 3.10+
    - PyTorch
    - Datasets (`pip install datasets`)
    - Tiktoken (`pip install tiktoken`)
    - Matplotlib (for plotting)

2. **Run Notebook Steps**  
    - Each notebook cell is annotated by step for clarity.
    - You may modify model hyperparameters (`n_layer`, `n_head`, `n_embd`, etc.) to change the model’s size.

3. **Train the Model**  
    - Training on full TinyStories may take several hours on consumer GPUs.
    - Checkpointing is automatic; best model parameters are saved to `best_model_params.pt`.

4. **Perform Inference**  
    - Enter a prompt such as `"Once upon a time there was a pumpkin."`
    - Observe model-generated continuation.

## Example

```
sentence = "A little girl went to the woods"
context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim = 0))
y = model.generate(context, 200)
print(enc.decode(y.squeeze().tolist()))
```

## Model Configuration

Default configuration:
- Vocab size: 50257 (GPT-2 tiktoken)
- Context window (`block_size`): 128 tokens
- Layers: 6
- Heads: 6
- Embedding size: 384
- Dropout: 0.1

All configurable via the `GPTConfig` dataclass in the notebook.

## Dataset Reference

- [TinyStories on HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories)

## Acknowledgements

- [nanoGPT](https://github.com/karpathy/nanoGPT) for streamlined GPT implementations.
- TinyStories dataset authors for the synthetic children's story corpus.
- HuggingFace for easy dataset access.

## License

This implementation is provided for educational and research purposes. Consult the TinyStories dataset license before re-distributing data or models trained on it.
