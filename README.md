# CT-GAN: Contrastive Text-to-Image Generation using SSA

This repository contains an implementation of the CT-GAN model, a GAN-based architecture for high-resolution text-to-image synthesis conditioned on natural language descriptions. This model extends StackGAN by integrating Contrastive Learning, Shift Self-Attention (SSA), and a multi-stage generation pipeline for 64x64, 128x128, and 256x256 image synthesis.

## 🧠 Key Features

* **Stage-wise GAN Architecture:**

  * Stage I: 64x64 coarse image generation
  * Stage II: 128x128 refinement with filtering
  * Stage III: 256x256 fine image synthesis with global/local discriminators

* **Shift Self-Attention (SSA):** Used in Generator for learning spatial features and improving visual details

* **Filtering Module:** Discards low-quality Stage I outputs before refining in higher stages

* **Char-CNN-RNN Embeddings:** Pretrained text embeddings used as conditioning input

* **CLIP Loss (optional):** Improves semantic alignment between text and image features

* **Checkpointing:** Model checkpoints saved every 50 epochs with full optimizer and network state

* **FID & Inception Score:** Evaluation-ready metrics for generated image quality


## 🚀 Training Instructions

1. **Preprocess Data:** Ensure CUB-200 dataset with bounding boxes and `text_c10` folder is set up.
2. **Stage I Training:**

```bash
python train.py --stage 1 --epoch 200 --z_dim 100 --lr_gen 0.0002 --lr_dis 0.0002
```

3. **Stage II Training:**

```bash
python train.py --stage 2 --epoch 200 --STAGE1_G /path/to/gen1.pth --STAGE1_D /path/to/dis1.pth
```

4. **Stage III Training:**

```bash
python train.py --stage 3 --epoch 200 --STAGE1_G /path/to/gen1.pth --STAGE1_D /path/to/dis1.pth
```

## 📊 Evaluation Metrics

* `FID`: Frechet Inception Distance between generated and real image distributions
* `IS`: Inception Score on test set outputs

Run evaluation after training:

```bash
python evaluate.py --model_path /path/to/model.pth --stage 3
```

## ✨ Sample Results

Output images during training are saved in `/output_images/epoch_X/batch_Y.png` and `epoch_X_final.png`. (Add `.png` examples here)

## 👨‍💻 Contributions (for Resume / Mentorship)

**Part 1: Data Loading + Stage I**

* Implemented custom PyTorch `ImageDataset` for loading CUB-200 and `text_c10` captions
* Trained Stage I GAN from scratch with char-CNN-RNN embeddings
* Integrated checkpoint saving, optimizer resuming, and noise conditioning

**Part 2: Stage II + Filtering**

* Developed Stage II Generator & Discriminator with SSA and Filtering module
* Loaded Stage I pretrained models and used filtering to reject poor samples
* Refined image quality using intermediate representations

**Part 3: Stage III + Evaluation**

* Implemented 256x256 Generator and Discriminator using SSA and residual upsampling
* Integrated CLIP-based loss for semantic alignment (optional)
* Added FID & Inception Score evaluation pipeline
* Fine-tuned models and visualized batch-wise image results

## 📌 Dependencies

* PyTorch
* torchvision
* numpy, pandas
* PIL
* scikit-learn
* OpenCV

## 📜 License

MIT License

---

> Feel free to fork, modify, and contribute to this repository.
