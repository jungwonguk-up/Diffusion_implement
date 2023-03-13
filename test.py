import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt

# Invertigating the dataset
def show_images(data, num_samples=20, cols=4): # cols 가 일반적으로 그냥 idx를 의미하는건가?
    """
    Plots some samples from the datset
    """
    plt.figure(figsize=(15,15))
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(num_samples/cols + 1, cols, i + 1)
        plt.imshow(img[0])

def data_viz():
    data = torchvision.datasets.StanfordCars(root=".", download=True)
    show_images(data)


"""
Buiding the Diffusion Modle
"""

# Step 1 : The forward process = Noise scheduler

def linear_beta_schedule(timesteps, start=0.001, end=0.02):
    return torch.linspace(start, end, timesteps)    # linspace가 뭐하는 거지?


def get_index_from_list(vals, t, x_shape):
    """
    Returns a aspecific index t of a passed list of values vals
    while considering the batich dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())  # gather이 뭐더라?
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)   # randn_like 가 뭐지?
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)








