import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim import AdamW
from diffusers.optimization import get_scheduler
from torch.cuda.amp import autocast, GradScaler

# Configuration
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATASET_SIZE = 300  # Using 500 images as specified
BATCH_SIZE = 4
NUM_EPOCHS = 2
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "fine_tuned_sd"
DATA_DIR = "/data/rashidm/COCO"  # Adjust this to your local dataset path if different

# Check GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("GPU not available. This script requires a CUDA-compatible GPU.")
print(DEVICE)

# Load and preprocess COCO dataset
def load_coco_subset():
    dataset = load_dataset("phiyodr/coco2017", split="train").flatten().select(range(DATASET_SIZE))
    return dataset

# Custom Dataset to handle tensor conversion
class COCODataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        if self.transform:
            image_path = os.path.join(DATA_DIR, example["file_name"])
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        text = tokenizer(
            example["captions"][0],  # Use first caption
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]  # Get tensor of shape [sequence_length]
        return {"processed_image": image, "text": text}

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Prepare dataset with custom Dataset class
dataset = load_coco_subset()
custom_dataset = COCODataset(dataset, transform=transform)
dataloader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=lambda x: {
    "processed_image": torch.stack([item["processed_image"] for item in x]),
    "text": torch.stack([item["text"] for item in x])
})

# Load Stable Diffusion components
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE)
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(DEVICE)
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(DEVICE)
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# Freeze VAE to save memory
vae.requires_grad_(False)

# Optimizer and scheduler
optimizer = AdamW(unet.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=len(dataloader) * NUM_EPOCHS
)

# Mixed precision scaler
scaler = GradScaler()

# Training loop
epsilon = 0.01

unet.train()
text_encoder.train()

for epoch in range(NUM_EPOCHS):
    for batch in dataloader:
        images = batch["processed_image"].to(DEVICE)
        text   = batch["text"].to(DEVICE)

        # === 1) FGSM attack pass ===
        images_adv = images.clone().detach().requires_grad_(True)

        with autocast():
            # 1a) embed text for attack
            text_embeds_adv = text_encoder(text).last_hidden_state

            # 1b) forward on images_adv
            latents = vae.encode(images_adv).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred = unet(noisy_latents, timesteps, text_embeds_adv).sample
            loss_adv = torch.nn.functional.mse_loss(noise_pred, noise)

        # get pixel gradients and craft FGSM image
        loss_adv.backward()
        images_fgsm = (images_adv + epsilon * images_adv.grad.sign())\
                        .detach().clamp(0, 1)

        # clear model grads (we only wanted imgs’ grads above)
        optimizer.zero_grad()
        images_adv.grad.zero_()

        # === 2) Fine‑tuning pass on adversarial images ===
        with autocast():
            # 2a) embed text *again* (fresh graph)
            text_embeds = text_encoder(text).last_hidden_state

            # 2b) forward on FGSM‑perturbed images
            latents = vae.encode(images_fgsm).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred = unet(noisy_latents, timesteps, text_embeds).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # update UNet weights
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS}  "
            f"FGSM-loss: {loss_adv.item():.4f}  "
            f"Train-loss: {loss.item():.4f}"
        )

# Save fine-tuned model
unet.save_pretrained(f"{OUTPUT_DIR}/unet")
text_encoder.save_pretrained(f"{OUTPUT_DIR}/text_encoder")
vae.save_pretrained(f"{OUTPUT_DIR}/vae")

# Inference example
pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    unet=UNet2DConditionModel.from_pretrained(f"{OUTPUT_DIR}/unet"),
    text_encoder=CLIPTextModel.from_pretrained(f"{OUTPUT_DIR}/text_encoder"),
    vae=AutoencoderKL.from_pretrained(f"{OUTPUT_DIR}/vae")
).to(DEVICE)

# Generate an image
image = pipeline("A dog playing in a park", num_inference_steps=50).images[0]
image.save("generated_image_adv.png")

from transformers import CLIPProcessor, CLIPModel

# 1) Initialize CLIP
clip_model   = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor    = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_clip_score(pipeline, prompt: str, steps: int = 50) -> float:
    """
    Generate an image with `pipeline` from `prompt`, then compute
    cosine similarity between CLIP image- and text-embeddings.
    """
    # a) Generate image
    image = pipeline(prompt, num_inference_steps=steps).images[0]

    # b) Preprocess for CLIP (batch of size 1)
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    # c) Forward through CLIP
    outputs = clip_model(**inputs)

    # d) logits_per_image is (batch_size, batch_size) similarity matrix
    #    here batch_size=1, so [0,0] is our cos‑sim score
    score = outputs.logits_per_image[0,0].item()
    return score

# 2) Example usage:
prompt     = "A dog playing in a park"
clean_score = compute_clip_score(pipeline, prompt)
adv_score   = compute_clip_score(pipeline, prompt)  # or use your adv‑fine‑tuned pipeline

print(f"Clean CLIP‑score: {clean_score:.4f}")
print(f"Attacked CLIP‑score: {adv_score:.4f}")
