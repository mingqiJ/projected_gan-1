# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, num_gpus=1):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Gmain = (-gen_logits).mean()

                # added for transitional training
                gen_logits_c = None
                if gen_logits.size(1) == 2:
                    gen_logits_c = gen_logits[:, 1]
                    gen_logits = gen_logits[:, 0]

                if gen_logits_c is not None:
                    training_stats.report('Loss/scores/fake_c', gen_logits_c)
                    training_stats.report('Loss/signs/fake_c', gen_logits_c.sign())
                    loss_Gmain_c = (-gen_logits_c).mean()

                    if num_gpus > 1:
                        loss_Gmain = loss_Gmain + self.D.module.transition * loss_Gmain_c
                    else:
                        loss_Gmain = loss_Gmain + self.D.transition * loss_Gmain_c

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                # added for transitional training
                gen_logits_c = None
                if gen_logits.size(1) == 2:
                    gen_logits_c = gen_logits[:, 1]
                    gen_logits = gen_logits[:, 0]

                if gen_logits_c is not None:
                    training_stats.report('Loss/scores/fake_c', gen_logits_c)
                    training_stats.report('Loss/signs/fake_c', gen_logits_c.sign())
                    loss_Dgen_c = (F.relu(torch.ones_like(gen_logits_c) + gen_logits_c)).mean()

                    if num_gpus > 1:
                        loss_Dgen = loss_Dgen + self.D.module.transition * loss_Dgen_c
                    else:
                        loss_Dgen = loss_Dgen + self.D.transition * loss_Dgen_c

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                # added for transitional training
                real_logits_c = None
                if real_logits.size(1) == 2:
                    real_logits_c = real_logits[:, 1]
                    real_logits = real_logits[:, 0]

                if real_logits_c is not None:
                    training_stats.report('Loss/scores/real_c', real_logits_c)
                    training_stats.report('Loss/signs/real_c', real_logits_c.sign())
                    loss_Dreal_c = (F.relu(torch.ones_like(real_logits_c) - real_logits_c)).mean()
                    if num_gpus > 1:
                        loss_Dreal = loss_Dreal + self.D.module.transition * loss_Dreal_c
                    else:
                        loss_Dreal = loss_Dreal + self.D.transition * loss_Dreal_c

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()
