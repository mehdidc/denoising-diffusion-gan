from score_sde.models.projected_discriminator import ProjectedDiscriminator
import torch
discr = ProjectedDiscriminator(num_discs=4, backbone_kwargs={"cond_size": 768})
x = torch.randn(1,3,224,224)
t = torch.randint(0, 1, size=(1,))
cond = (None, torch.randn(1,77, 768), torch.ones(1,77, dtype=torch.bool))
y = discr(x, t, x, cond=cond)
print(y.shape)
