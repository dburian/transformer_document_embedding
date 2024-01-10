# How to create custom HuggingFace models?

At the end I decided to rely solely on `torch.nn.Module`. This got rid of a lot
of HF code, which IMHO is great for storing the model, learning about it and
quickly using it but not for experimentation. This is because to write HF models
one needs to do a lot of copying and writing very verbose code.

So the steps are:
- download HF model
- put it inside custom `torch.nn.Module`
- work only with the custom module
- save only the state dict
