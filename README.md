# Russian Jokes Mini LLM

End-to-end notebook project for training a compact causal language model for Russian joke generation. The project includes a custom Byte-level BPE tokenizer, a decoder-only Transformer implementation, model training, hyperparameter search, architecture experiments, generation examples, and Hugging Face Hub export utilities.

## Highlights

- Trains a Byte-level BPE tokenizer with a 1024-token vocabulary.
- Implements a decoder-only Transformer with ALiBi, Grouped-Query Attention, RMSNorm, and SwiGLU.
- Compares `nano`, `mini`, and `small` model configurations.
- Runs refined hyperparameter search for the stronger configurations.
- Adds RoPE and Multi-Head Latent Attention variants for architecture comparison.
- Keeps executed notebook outputs so results can be reviewed without rerunning the full training pipeline.

## Results

The saved run in the notebook reports these best validation losses on the holdout split:

| model | parameters | steps | val_loss |
|---|---:|---:|---:|
| `mini` | 10.52M | 10,000 | 2.8547 |
| `small` | 79.45M | 12,000 | 2.7377 |

The lightweight architecture comparison on the `nano` setup showed `TransformerForCausalLMMLA` outperforming the RoPE variant in the short 700-step run (`4.9573` vs `5.0737` validation loss).

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks
    └── russian_jokes_mini_llm.ipynb
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/russian_jokes_mini_llm.ipynb
```

Full training is GPU-oriented. The compact experiments are suitable for Colab, Kaggle, or a local CUDA setup.

## Hugging Face Export

The notebook is configured to publish the tokenizer, model, and generated model card to a Hub repository named `russian-jokes-mini-llm` under the authenticated Hugging Face account. The model card is generated as `hf_model_card.md` locally and uploaded as `README.md` to the model repository.

Model weights are not committed to this Git repository; they should be trained locally or stored on Hugging Face Hub.
