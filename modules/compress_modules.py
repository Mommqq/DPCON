import torch.nn as nn
from .network_components import ResnetBlock, VBRCondition, FlexiblePrior, Downsample, Upsample, GDN1
from modules.cross_attention import CrossAttention
from .utils import quantize, NormalDistribution
import torch

class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
        image_size = (128,256)
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)] 
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))  
        self.reversed_dims = list(reversed([out_channels, *map(lambda m: dim * m, dim_mults)])) 
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:])) 
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)] 
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:])) 
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        )
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:]) 
        )
        self.vbr = vbr
        self.prior_x = FlexiblePrior(self.hyper_dims[-1])
        self.prior_y = FlexiblePrior(self.hyper_dims[-1])

        self.ca1 = CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=self.dims[3],
                                  dim=256, num_patches=4, heads=8, dropout=0.1)
        self.ca2 = CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=self.dims[2],
                                  dim=256, num_patches=4, heads=8, dropout=0.1)
        self.ca3 = CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=self.dims[1],
                                  dim=256, num_patches=4, heads=8, dropout=0.1)
        self.attention = nn.ModuleList([])
        self.attention.append(self.ca1)
        self.attention.append(self.ca2)
        self.attention.append(self.ca3)

    def get_extra_loss(self):
        return self.prior_x.get_extraloss()+self.prior_y.get_extraloss()

    def build_network(self):
        self.enc_x = nn.ModuleList([])
        # self.enc_w = nn.ModuleList([])
        self.enc_y = nn.ModuleList([])
        self.dec_x = nn.ModuleList([])       
        self.dec_y = nn.ModuleList([])
        self.hyper_enc_x = nn.ModuleList([])
        self.hyper_dec_x = nn.ModuleList([])
        self.hyper_enc_y = nn.ModuleList([])
        self.hyper_dec_y = nn.ModuleList([])
        # self.hyper_enc_w = nn.ModuleList([])
        # self.hyper_dec_w = nn.ModuleList([])

    def encode_x(self, input, cond=None):
        for i, (resnet, vbrscaler, down) in enumerate(self.enc_x):
            input = resnet(input)
            if self.vbr:
                input = vbrscaler(input, cond)
            input = down(input)
        latent = input
        for i, (conv, vbrscaler, act) in enumerate(self.hyper_enc_x):
            input = conv(input)
            if self.vbr and i != (len(self.hyper_enc_x) - 1):
                input = vbrscaler(input, cond)
            input = act(input)
        hyper_latent = input
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior_x.medians)
        input = q_hyper_latent
        for i, (deconv, vbrscaler, act) in enumerate(self.hyper_dec_x):
            input = deconv(input)
            if self.vbr and i != (len(self.hyper_dec_x) - 1):
                input = vbrscaler(input, cond)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return q_latent, q_hyper_latent, state4bpp

    def encode_y(self, input, cond=None):
        for i, (resnet, vbrscaler, down) in enumerate(self.enc_y):
            input = resnet(input)
            if self.vbr:
                input = vbrscaler(input, cond)
            input = down(input)
        latent = input
        for i, (conv, vbrscaler, act) in enumerate(self.hyper_enc_y):
            input = conv(input)
            if self.vbr and i != (len(self.hyper_enc_w) - 1):
                input = vbrscaler(input, cond)
            input = act(input)
        hyper_latent = input
        # q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
       
        for i, (deconv, vbrscaler, act) in enumerate(self.hyper_dec_y):
            input = deconv(input)
            if self.vbr and i != (len(self.hyper_dec_y) - 1):
                input = vbrscaler(input, cond)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        # q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return latent, hyper_latent, state4bpp


    def decode(self, x, y, cond=None):
        output = []
        output_y = []
        x = torch.cat((x,y),1)
        # y = torch.cat((y,w),1)
        for i, ((resnet_x, vbrscaler_x, down_x), (resnet_y, vbrscaler_y, down_y)) in enumerate(zip(self.dec_x, self.dec_y)):

            x = resnet_x(x)
            if self.vbr:
                x = vbrscaler_x(x, cond)
            x = down_x(x)
            y = resnet_y(y)
            if self.vbr:
                y = vbrscaler_y(y, cond)
            y = down_y(y) 
                      
            output.append(x)
            output_y.append(y)
            x = torch.cat((x,y),1) 
            # if i<=2:
            #     x = self.attention[i](x,y)

        return output[::-1],output_y[::-1]

    def bpp(self, shape, state4bpp_x, state4bpp_y):
        B, _, H, W = shape
        
        latent_x = state4bpp_x["latent"]
        hyper_latent_x = state4bpp_x["hyper_latent"]
        latent_distribution_x = state4bpp_x["latent_distribution"]

        latent_y = state4bpp_y["latent"]
        hyper_latent_y = state4bpp_y["hyper_latent"]
        latent_distribution_y = state4bpp_y["latent_distribution"]

        if self.training:
            q_hyper_latent_x = quantize(hyper_latent_x, "noise")
            q_latent_x = quantize(latent_x, "noise")

        else:
            q_hyper_latent_x = quantize(hyper_latent_x, "dequantize", self.prior_x.medians)
            q_latent_x = quantize(latent_x, "dequantize", latent_distribution_x.mean)

        hyper_rate_x = -self.prior_x.likelihood(q_hyper_latent_x).log2()
        cond_rate_x = -latent_distribution_x.likelihood(q_latent_x).log2()
        bpp_x = (hyper_rate_x.sum(dim=(1, 2, 3)) + cond_rate_x.sum(dim=(1, 2, 3))) / (H * W)

        hyper_rate_y = -self.prior_y.likelihood(hyper_latent_y).log2()
        cond_rate_y = -latent_distribution_y.likelihood(latent_y).log2()
        bpp_y = (hyper_rate_y.sum(dim=(1, 2, 3)) + cond_rate_y.sum(dim=(1, 2, 3))) / (H * W)
        return bpp_x,bpp_y

    def forward(self, input_x,input_y, cond=None):
        q_latent, q_hyper_latent, state4bpp = self.encode_x(input_x, cond)
        latent_y, hyper_latent_y, state4bpp_y = self.encode_y(input_y, cond) 
        # latent_w, hyper_latent_w, state4bpp_w = self.encode_w(input_y, cond)      
        bpp_x,bpp_y = self.bpp(input_x.shape, state4bpp, state4bpp_y)
        output,output_y = self.decode(q_latent, latent_y, cond)
        # print(y.shape)
        return {
            "output": output,
            "bpp": bpp_x,
            "bpp_y":bpp_y,
            # "bpp_w":bpp_w,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
            "output_y":output_y
        }


class BigCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 3, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
    ):
        super().__init__(dim, dim_mults, hyper_dims_mults, channels, out_channels, vbr)
        self.build_network()

    def build_network(self):

        self.enc_x = nn.ModuleList([])
        self.enc_w = nn.ModuleList([])
        self.enc_y = nn.ModuleList([])
        self.dec_x = nn.ModuleList([])       
        self.dec_y = nn.ModuleList([])
        self.hyper_enc_x = nn.ModuleList([])
        self.hyper_dec_x = nn.ModuleList([])
        self.hyper_enc_y = nn.ModuleList([])
        self.hyper_dec_y = nn.ModuleList([])
        self.hyper_enc_w = nn.ModuleList([])
        self.hyper_dec_w = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc_x.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        VBRCondition(1, dim_out) if self.vbr else nn.Identity(),
                        Downsample(dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc_y.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        VBRCondition(1, dim_out) if self.vbr else nn.Identity(),
                        Downsample(dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            in0 = ind==0            
            self.dec_x.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in*2, dim_out if not is_last else dim_in),
                        # ResnetBlock(dim_in*2 if in0 else dim_in, dim_out if not is_last else dim_in),                        
                        VBRCondition(1, dim_out if not is_last else dim_in)
                        if self.vbr
                        else nn.Identity(),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            in0 = ind==0
            self.dec_y.append(
                nn.ModuleList(
                    [
                        # ResnetBlock(dim_in*2 if in0 else dim_in, dim_out if not is_last else dim_in),
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        VBRCondition(1, dim_out if not is_last else dim_in)
                        if self.vbr
                        else nn.Identity(),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc_x.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec_x.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc_y.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec_y.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc_w.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec_w.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )




class SimpleCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
    ):
        super().__init__(dim, dim_mults, hyper_dims_mults, channels, out_channels, vbr)
        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        GDN1(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        GDN1(dim_out, True) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )
