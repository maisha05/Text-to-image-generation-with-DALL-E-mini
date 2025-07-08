import os
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad

from dalle_mini import DalleBart
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPTokenizer
from datasets import load_dataset
from PIL import Image
import optax
import matplotlib.pyplot as plt

# --- 1) Load & preprocess 500 COCO samples ---

print("Loading COCO (500 samples)…")
ds = load_dataset("phiyodr/coco2017", split="train").flatten().select(range(500))

IMG_DIR = "/data/rashidm/COCO"
def add_path(ex):
    ex["image_path"] = os.path.join(IMG_DIR, ex["file_name"])
    return ex
ds = ds.map(add_path)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
def preprocess(ex):
    img = Image.open(ex["image_path"]).convert("RGB").resize((256,256))
    cap = ex["captions"][0] if isinstance(ex["captions"], list) else ex["captions"]
    toks = tokenizer(cap,
                     padding="max_length", max_length=256,
                     truncation=True, return_tensors="np")
    return {
        "input_ids": toks["input_ids"].squeeze(0),
        "image": np.array(img, dtype=np.float32)/255.0
    }
print("Preprocessing images & captions…")
ds = ds.map(preprocess)
ds.set_format(type="numpy", columns=["input_ids","image"])

data      = ds[:]
input_ids = np.stack(data["input_ids"],axis=0).astype(np.int32)  # (500,256)
images    = np.stack(data["image"],    axis=0).astype(np.float32) # (500,256,256,3)

# --- 2) Load models & encode images ---

print("Loading DALL·E mini + VQGAN…")
model = DalleBart.from_pretrained("dalle-mini/dalle-mini")
vqvae = VQModel.from_pretrained("dalle-mini/vqgan_imagenet_f16_16384")

# encode images in batches
batch_enc = 16
num_enc   = math.ceil(len(images)/batch_enc)
all_tokens = []
print("Encoding images through VQGAN…")
for i in range(num_enc):
    s,e = i*batch_enc, min((i+1)*batch_enc, len(images))
    toks = vqvae.encode(jnp.array(images[s:e]))[1]     # (batch,256)
    all_tokens.append(np.array(toks))
image_tokens = np.concatenate(all_tokens,axis=0)  # (500,256)

# Verify token indices
print("Max image token index =", image_tokens.max())
print("Min image token index =", image_tokens.min())

# Use fixed codebook size
codebook_size = 16384  # Fixed for vqgan_imagenet_f16_16384
print("Using fixed codebook size =", codebook_size)

# Validate image tokens
if image_tokens.max() >= codebook_size:
    raise ValueError(f"Image tokens contain indices >= {codebook_size}, which exceeds the codebook size.")
if image_tokens.min() < 0:
    raise ValueError("Image tokens contain negative indices, which is invalid.")

# --- 3) Prepare model params and optimizer ---

params    = {"model": model.params}
optimizer = optax.adam(1e-5)
opt_state = optimizer.init(params)
decoder_start_token_id = model.config.decoder_start_token_id

# --- 4) Loss & train step ---

# Before training, outside JIT:
print("Max target token:", image_tokens.max())
print("Min target token:", image_tokens.min())
assert image_tokens.max() < codebook_size
assert image_tokens.min() >= 0

# Inside loss_fn (no assert!):
def loss_fn(params, batch_in, batch_tgt):
    x = jnp.array(batch_in)
    bs, seq = x.shape
    start = jnp.full((bs,1), decoder_start_token_id, dtype=x.dtype)
    decoder_ids = jnp.concatenate([start, x[:,:-1]],axis=1)

    out   = model(input_ids=x,
                  decoder_input_ids=decoder_ids,
                  params=params["model"],
                  train=False)
    lgts  = jnp.clip(out.logits, -5.0, 5.0)   # (B,T,16385)

    codebook_size = lgts.shape[-1]  # 16385
    # No assert here!
    loss = optax.softmax_cross_entropy(
        lgts,
        jax.nn.one_hot(batch_tgt, codebook_size)
    ).mean()
    return loss

@jax.jit
def train_step(params, opt_state, b_in, b_tgt):
    (l, grads) = value_and_grad(loss_fn)(params, b_in, b_tgt)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, l

# --- 5) Fine-tune loop ---

print("Starting fine-tuning…")
epochs = 2
bs     = 4
steps  = len(input_ids)//bs

for e in range(1, epochs+1):
    total = 0.0
    for i in range(steps):
        bi = input_ids[i*bs:(i+1)*bs]
        ti = image_tokens[i*bs:(i+1)*bs]
        params, opt_state, l = train_step(params, opt_state, bi, ti)
        total += l
        print(f"Epoch {e} • Batch {i+1}/{steps} • Loss {l:.4f}")
        if jnp.isnan(l):
            print("NaN encountered, aborting.")
            break
    print(f"→ Epoch {e} avg loss {total/steps:.4f}")

# save
model.params = params["model"]
model.save_pretrained("/data/rashidm/fine_tuned_dalle_mini")
print("Saved fine-tuned model.")

# --- 6) Quick demo generation ---

cap = "A dog playing in a park"
tok = tokenizer(cap,
                padding="max_length", max_length=256,
                truncation=True, return_tensors="np")["input_ids"]

with jax.default_matmul_precision("float32"):
    x = jnp.array(tok)
    start = jnp.full((x.shape[0],1), decoder_start_token_id, dtype=x.dtype)
    dec = jnp.concatenate([start, x[:,:-1]],axis=1)

    out = model(input_ids=x,
                decoder_input_ids=dec,
                params=model.params, train=False)
    lg = jnp.clip(out.logits, -5.0, 5.0)
    # Slice if needed for demo as well
    if lg.shape[-1] != codebook_size:
        print(f"Warning: Model output vocab {lg.shape[-1]} != codebook_size {codebook_size}, slicing logits.")
        lg = lg[..., :codebook_size]
    gi = jnp.argmax(lg,axis=-1)[0]  # (256,)

arr = np.array(gi)
sz  = int(math.sqrt(arr.size))
img = (arr.reshape(sz,sz)*255//(codebook_size-1)).astype("uint8")
Image.fromarray(img,mode="L").save("generated_demo.png")
plt.imshow(img, cmap="gray"); plt.axis("off")
print("Demo saved as generated_demo.png")
