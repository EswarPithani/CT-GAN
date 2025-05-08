CT-GAN: Multi-Stage Text-to-Image Synthesis using Conditional GANs
This project implements CT-GAN, a multi-stage Conditional GAN for generating realistic images from natural language descriptions. Inspired by StackGAN and AttnGAN, the model progressively enhances image quality from low (64×64) to high resolution (256×256), conditioned on char-CNN-RNN text embeddings.

📌 Features
✅ Three-Stage Generator–Discriminator Architecture (64×64 → 128×128 → 256×256)

✍️ Text-Conditioning using pre-trained char-CNN-RNN sentence embeddings

🎯 Progressive Training with filtering modules for semantic refinement between stages

💾 Checkpointing support for resume training from any epoch and stage

📷 Output Visualization using torchvision and IPython.display

📊 Ready for FID / IS Score Evaluation and inference on custom text prompts

🧱 Model Architecture
Stage I: Generates low-resolution (64×64) images from noise + text

Stage II: Refines to 128×128 using previous stage + filtered output

Stage III: Further refines to 256×256 for photo-realism

🧪 Training
# Stage 1
python train.py --STAGE 1 --epoch 100 --NET_G '' --NET_D ''

# Stage 2 (uses pretrained Stage 1)
python train.py --STAGE 2 --STAGE1_G path/to/stage1_G.pth --STAGE1_D path/to/stage1_D.pth

# Stage 3 (uses pretrained Stage 2)
python train.py --STAGE 3 --STAGE1_G path/to/stage2_G.pth --STAGE1_D path/to/stage2_D.pth

🛠️ Tech Stack
PyTorch, Torchvision

char-CNN-RNN Embeddings

PIL, IPython for visualization

Trained on: Kaggle Notebooks

📈 Results
Generated samples at 256×256 conditioned on class-level descriptions from the CUB-200 dataset. Training pipeline supports extension to other datasets as well.

🤝 Contributions
This project was structured in three stages of contributions:

Stage-I (64×64): Designed and trained initial text-conditioned GAN from scratch

Stage-II (128×128): Integrated intermediate refinement with filtered Stage-I output

Stage-III (256×256): Implemented final image enhancement and model checkpointing

