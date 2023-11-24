import torch
import ddpm
import ddpm_conditional
import modules
import utils
# import noising_test
def sample(conditional):
    if conditional==False:
        device = "cuda"
        model = modules.UNet().to(device)
        ckpt = torch.load("./models/unconditional_ckpt.pt")
        model.load_state_dict(ckpt)
        diffusion = ddpm.Diffusion(img_size=64, device=device)
        x = diffusion.sample(model, n=16)
        utils.plot_images(x)
    else:
        n = 10
        device = "cuda"
        model = modules.UNet_conditional(num_classes=10).to(device)
        ckpt = torch.load("./models/conditional_ckpt.pt")
        model.load_state_dict(ckpt)
        diffusion = ddpm_conditional.Diffusion(img_size=64, device=device)
        y = torch.Tensor([6] * n).long().to(device)
        x = diffusion.sample(model, n, y, cfg_scale=3)
        utils.plot_images(x)

if __name__ == "__main__":
    sample(conditional=True)