# Code style guide

This guide describes how I write code. It is based on my own past work: a domain-adaptation training script (`train.py`), two method files (`dan.py`, `sam.py`), and a ViT notebook (`Task2_ViT.ipynb`).

**For Claude Code:** follow these rules for all new code in this repo. Follow them for any edits to my own code too. Leave the structure of course-provided starter files alone, and apply this style only to the parts you add.

---

## 1. Overall philosophy

- **Simple and explicit over clever.** Use plain functions, plain loops and plain lists. Don't build abstractions such as class hierarchies, registries, decorators, dataclasses or config frameworks, except for an `nn.Module` where PyTorch needs one.
- **Readable top to bottom.** A reader should be able to follow a script in order without jumping between many files.
- **One method per file.** Each training method lives in its own module, for example `methods/dan.py` or `methods/sam.py`. It exposes one `<method>_train(...)` function that does **one optimisation step**, plus a few small helpers above it.
- **Some branching is fine.** In the main loop it's fine to have one `if/elif` branch per method, even when the branches look alike. Each branch stays readable on its own.

## 2. File layout

Order the contents of a module like this:

1. Imports.
2. Module-level constants in `UPPER_CASE`, such as `SEED=6304` or `SOURCE_DOMAINS`.
3. Small helper functions.
4. The main public function, such as `dan_train` or `main`.
5. In entry-point scripts, an `if __name__ == "__main__":` block that builds the argparse parser and calls `main(args)`.

## 3. Imports

- Put `import x` lines first (`os`, `json`, `random`, `argparse`, `numpy as np`, `torch`). Then put `from project.module import thing` lines.
- Use the common aliases: `import numpy as np`, `import torch.nn.functional as F`, and `import matplotlib.pyplot as plt`.
- Use absolute imports from the project root, such as `from task2.models.backbone import ResNetBackbone`.
- Put all imports at the top, even though some of my old code added imports later.

## 4. Naming

- Use `snake_case` for variables and functions, and `CamelCase` for model classes (`ResNetBackbone`, `ClassifierHead`, `DomainDiscriminator`).
- **Short abbreviations are normal:** `src`, `tgt` or `target`, `feats`, `imgs`, `cls_loss`, `tot_loss`, `bw`, `atten`, `preds`, `k_ss`, `k_st`, `e_w`.
- Name functions with verbs or nouns that say what they compute: `rbf_kernel`, `median_dist`, `mmd_loss`, `source_loss`, `get_perturbs`, `evaluate_model`.
- Build run names from the hyperparameter being varied, for example `run_name = f"dan_lambda_{args.lambda_mmd}"`.

## 5. Formatting

- I often write assignments as `name= value`, with a space after `=` but not before it. Mixing this with `name = value` is fine, so don't reformat existing lines just to make them consistent.
- I use lots of blank lines inside functions to separate small logical steps. Often there is one between almost every statement.
- When a call doesn't fit on one line, put **one argument per line**:

  ```python
  scores = evaluate_model(
      backbone,
      classifier,
      source_val[domain],
      device
  )
  ```

- Call training-step functions with **keyword arguments**, one per line:

  ```python
  total_loss, cls_loss, alignment_loss = dan_train(
      backbone=backbone,
      classifier=classifier,
      src_iter=source_iters,
      src_loader=source_train,
      target_imgs=target_imgs,
      optimizer=optimizer,
      criterion=criterion,
      device=device,
      lambda_mmd=args.lambda_mmd
  )
  ```

- Use 4-space indentation and double quotes for strings.
- Don't run an auto-formatter such as black or ruff over the code.

## 6. Comments, docstrings and types

- **No docstrings** and **no type hints**.
- Use few comments. Write them lowercase, often with no space after `#`. A comment explains a decision or a shape, not what the code obviously does:

  ```python
  #decided to use huggingface ViT-Base/16
  #need to go from 14x14 to 224x224 as ViT input is 224x224
  ```

- An inline comment is fine to label a magic index, as in `image_idx= 0   #0= polar bear`.

## 7. Scripts and configuration

- Entry scripts use `argparse`. Each `add_argument` is spread over several lines with `type=`, `default=` and, where useful, `choices=`:

  ```python
  parser.add_argument(
      "--method",
      type=str,
      choices=["source_only", "dan", "dann", "cdan"],
      default="source_only"
  )
  ```

- Argument names use underscores, such as `--output_dir`, `--pacs_root` and `--lambda_mmd`.
- Expose the hyperparameters an experiment varies as arguments. Fixed training constants can be plain variables near the top of `main`, for example `patience=5`. **In this repo, the assignment's fixed values come from `configs/`.**
- Seed everything at the start of `main`:

  ```python
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)

  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(SEED)
  ```

- Set up the device with `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, then `print("Using device:", device)`.
- Create the output folder with `os.makedirs(args.output_dir, exist_ok=True)` and build paths with `os.path.join`.

## 8. Training code patterns

- **A step function** puts the models in `.train()` mode, moves the batch to the device, calls `optimizer.zero_grad()`, computes the losses, then calls `backward()` and `optimizer.step()`. It returns the losses as **plain floats via `.item()`**, as a tuple:

  ```python
  return (tot_loss.item(), cls_loss.item(), alignment_loss.item())
  ```

- **Combine losses** explicitly, for example `tot_loss= cls_loss+ lambda_mmd*alignment_loss`.
- **Refill an iterator** when it runs out:

  ```python
  try:
      target_imgs, _ = next(target_iter)
  except StopIteration:
      target_iter = iter(target_train)
      target_imgs, _ = next(target_iter)
  ```

- **Track metrics** by appending to one Python list per metric (`losses`, `cls_losses` and so on). Average them at the end of the epoch with `sum(x) / len(x)`.
- **Write the math out by hand** with tensor operations rather than calling a library helper, as in `rbf_kernel` and `median_dist`. Clamp or add a small epsilon for numerical safety (`+ 1e-12`, `.clamp(min=1e-6)`).
- Wrap in-place parameter updates in `with torch.no_grad():`.
- Write schedules as inline formulas, for example `alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0`.

## 9. Logging and outputs

- **Use `print` only**, with no `logging` module and no tqdm:
  - Mark each epoch with `print(f"\n--- Epoch {epoch + 1} ---")`.
  - Print scalars with a comma and rounding, as in `print("classification loss:", round(avg_cls_loss, 4))`.
  - Print multi-value lines as split f-strings with `:.4f`:

    ```python
    print(
        f"{domain}: "
        f"accuracy={acc:.4f}, "
        f"macro_f1={f1:.4f}"
    )
    ```

- **Keep a history** by building one `dict` per epoch, appending it to a `history` list, and **rewriting `<run_name>_history.json` every epoch** with `json.dump(history, f, indent=2)`.
- **Save checkpoints** as a dict of `state_dict()`s plus the epoch and best metric, saved with `torch.save` to `<run_name>_best.pt`.
- **Use early stopping** on the validation metric, with an `epochs_noimprov` counter and `patience`. Print `"no improvement (k/patience)"` and `"stopping early"`.

## 10. Notebooks

- Use markdown headers that match the handout's numbering, such as `### 2.1: Using a Pre-trained ViT`.
- Keep cells small, one step per cell, and print tensor shapes after each transformation.
- Make plots with `fig, axes = plt.subplots(1, n, figsize=(...))`, giving each axis `.set_title(...)` and `.axis("off")`, then `plt.tight_layout()` and `plt.show()`.
- Follow each result with a short markdown cell stating what happened, in one or two plain sentences.
- **In this repo, notebooks are only thin launchers.** All experiment logic lives in `.py` scripts.

## 11. Habits from my old code *not* to copy

- **Typos in names:** write `max_epochs`, not `max_epoc`.
- **Duplicated logic:** compute shared values such as the `alpha` schedule once, or in a small helper, instead of in each branch.
- **Repeated conditions:** don't write the same `if` twice in a row.
- **Late imports:** don't put imports in the middle of a file or notebook. Import each module once.
- **Odd-looking redundancy,** such as `.clamp(min=1e-6).clamp(min=0)`. Keep one clamp.
- **Arbitrary blank-line runs:** at most one blank line inside a function and two between top-level definitions.

## 12. Additions specific to this assignment (PA2)

These additions go beyond my usual style, but they're needed here:

- Every results JSON also records the **git commit hash, config, seed and GPU name**.
- Training scripts take **`--smoke`**, which runs a few steps on a tiny model on CPU, and **`--resume`**, which continues from the latest checkpoint in `outputs/`.
- Each objective has a **small unit test** on hand-made tensors, kept in `tests/` and written in the same plain style.

---

### Short example in this style

```python
import torch
import torch.nn.functional as F

def sequence_logps(model, input_ids, attention_mask, labels):
    logits= model(input_ids=input_ids, attention_mask=attention_mask).logits

    #shift so token t predicts token t+1
    logits= logits[:, :-1]
    labels= labels[:, 1:]

    mask= (labels != -100).float()

    #-100 isnt a valid index for gather, masked out below anyway
    safe_labels= labels.clamp(min=0)

    logps= torch.gather(F.log_softmax(logits, dim=-1), 2, safe_labels.unsqueeze(2)).squeeze(2)

    return (logps * mask).sum(dim=1)

def dpo_train(policy, ref_model, batch, optimizer, device, beta=0.1):
    policy.train()

    optimizer.zero_grad()

    chosen_logps= sequence_logps(
        policy,
        batch["chosen_ids"].to(device),
        batch["chosen_mask"].to(device),
        batch["chosen_labels"].to(device)
    )

    ...

    loss.backward()
    optimizer.step()

    return (loss.item(), reward_margin.item(), reward_acc.item())
```
