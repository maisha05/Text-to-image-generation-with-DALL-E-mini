import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from datasets import load_dataset
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
from torch.optim import AdamW
from diffusers.optimization import get_scheduler
from torch.cuda.amp import autocast, GradScaler

# ─── CONFIG ────────────────────────────────────────────────────────────────
MODEL_ID      = "runwayml/stable-diffusion-v1-5"
DATA_DIR      = "/data/rashidm/COCO"
DATASET_SIZE  = 5000
BATCH_SIZE    = 4
NUM_EPOCHS    = 3a
LR            = 1e-5
EPS           = 0.05   # FGSM magnitude
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_CLEAN  = "clean_sd_pipeline"
OUTPUT_PERT   = "pert_sd_pipeline"

if not torch.cuda.is_available():
    raise RuntimeError("GPU not available. This script requires a CUDA-compatible GPU.")
print("Using device:", DEVICE)

# ─── DATASET CLASS ─────────────────────────────────────────────────────────
class DatasetFromHF(Dataset):
    """Converts HF COCO subset to (img_tensor, text_ids) pairs."""
    def __init__(self, hf_ds, root, size=(512,512)):
        self.ds    = hf_ds
        self.root  = root
        self.tf    = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])
        self.token = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex  = self.ds[i]
        img = Image.open(os.path.join(self.root, ex["file_name"])).convert("RGB")

        img = self.tf(img)
        txt = self.token(
            ex["captions"][0],
            max_length=77,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).input_ids[0]
        return {"img": img, "txt": txt}

# ─── FGSM PERTURBATION HELPER ──────────────────────────────────────────────
def fgsm_perturb(images, text, unet, text_encoder, vae, scheduler, eps=EPS):
    """
    Run one FGSM step in image space to produce adversarial images.
    """
    images_adv = images.clone().detach().requires_grad_(True)
    # forward pass
    te    = text_encoder(text).last_hidden_state
    lat   = vae.encode(images_adv).latent_dist.sample() * 0.18215
    noise = torch.randn_like(lat)
    t     = torch.randint(
        0, scheduler.num_train_timesteps,
        (lat.shape[0],), device=DEVICE
    )
    noisy = scheduler.add_noise(lat, noise, t)
    pred  = unet(noisy, t, te).sample
    loss  = torch.nn.functional.mse_loss(pred, noise)
    # backprop to get gradients on images_adv
    loss.backward()
    # FGSM step
    images_fgsm = (images_adv + eps * images_adv.grad.sign()).detach().clamp(0,1)
    images_adv.grad.zero_()
    return images_fgsm

# ─── TRAIN PIPELINE FUNCTION ───────────────────────────────────────────────
def train_pipeline(perturb=False):
    # Load and subset dataset
    raw_ds = load_dataset("phiyodr/coco2017", split="train").flatten().select(range(DATASET_SIZE))
    ds     = DatasetFromHF(raw_ds, DATA_DIR)
    dl     = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        collate_fn=lambda b: {
            "img": torch.stack([x["img"] for x in b]),
            "txt": torch.stack([x["txt"] for x in b])
        }
    )

    # Load model components
    unet         = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE)
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(DEVICE)
    vae          = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(DEVICE)
    scheduler    = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    vae.requires_grad_(False)

    optimizer = AdamW(unet.parameters(), lr=LR)
    lr_sched  = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(dl)*NUM_EPOCHS
    )
    scaler = GradScaler()
    unet.train(); text_encoder.train()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for batch in dl:
            imgs = batch["img"].to(DEVICE)
            txts = batch["txt"].to(DEVICE)

            if perturb:
                imgs = fgsm_perturb(imgs, txts, unet, text_encoder, vae, scheduler)

            # fine-tune step
            with autocast():
                te = text_encoder(txts).last_hidden_state
                lat = vae.encode(imgs).latent_dist.sample() * 0.18215
                noise = torch.randn_like(lat)
                t = torch.randint(
                    0, scheduler.num_train_timesteps,
                    (lat.shape[0],), device=DEVICE
                )
                noisy = scheduler.add_noise(lat, noise, t)
                pred = unet(noisy, t, te).sample
                loss = torch.nn.functional.mse_loss(pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()
            optimizer.zero_grad()

        print(f"{'PERT' if perturb else 'CLEAN'} epoch {epoch+1}/{NUM_EPOCHS} – loss {loss.item():.4f}")

    # Save full pipeline
    out_dir = OUTPUT_PERT if perturb else OUTPUT_CLEAN
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler
    ).to(DEVICE)
    pipe.save_pretrained(out_dir)
    return pipe

# ─── RUN: Clean vs. Perturbed Training ─────────────────────────────────────
clean_pipe = train_pipeline(perturb=False)
pert_pipe  = train_pipeline(perturb=True)

# ─── EVALUATION ────────────────────────────────────────────────────────────
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clip_score(pipe, prompt, seed=42):
    gen = torch.Generator(DEVICE).manual_seed(seed)
    img = pipe(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=gen
    ).images[0]
    inp = processor(
        text=[prompt], images=[img],
        return_tensors="pt", padding=True
    ).to(DEVICE)
    return clip_model(**inp).logits_per_image[0,0].item()

prompt = "A dog playing in a park"
print("Clean CLIP-score: %.4f" % clip_score(clean_pipe, prompt))
print("Perturbed-trained CLIP-score: %.4f" % clip_score(pert_pipe, prompt))
