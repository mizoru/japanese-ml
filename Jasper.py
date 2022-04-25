from fastai.core.all import *

class oldJasperSubblock(Module):
    r"A basic building sub-block of Jasper"
    
    def __init__(self, in_channels, out_channels, kernel, stride, drop = 0.2, **kwargs):
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride, (kernel-1)//2, **kwargs)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        
    def forward(self, x, res = 0):
        x = self.conv(x)
        x = self.norm(x)
        if not isinstance(res, int): x += res
        x = self.act(x)
        return self.drop(x)
    
    
class JasperSubblock(Module):
    r"A basic building sub-block of Jasper with QuartzNet separable convolutions"
    
    def __init__(self, in_channels, out_channels, kernel, stride, drop = 0.2, **kwargs):
        self.conv = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel, stride, (kernel-1)//2, groups=in_channels, **kwargs),
                                 nn.Conv1d(in_channels, out_channels, 1, 1))
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        
    def forward(self, x, res = 0):
        x = self.conv(x)
        x = self.norm(x)
        if not isinstance(res, int): x += res
        x = self.act(x)
        return self.drop(x)
    
class JasperBlock(Module):
    
    def __init__(self, in_channels, out_channels, kernel, num_subblocks=3, drop = 0.2):
        self.subblocks = nn.Sequential(*[JasperSubblock(in_channels, out_channels, kernel, 1, drop = 0.2)] \
        + [JasperSubblock(out_channels, out_channels, kernel, 1, drop = 0.2) for i in range(num_subblocks-2)])
        self.res_subblock = JasperSubblock(out_channels, out_channels, kernel, 1, drop = 0.2)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.res_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x, res = self.subblocks(x), self.res_conv(x)
        x = self.res_subblock(x, self.res_norm(x))
        return x
    
class Jasper(Module):
    
    def __init__(self, n_mels, n_vocab, num_blocks_factor=1, num_subblocks = 3):
        KERNELS = [11, 13, 17, 21, 25]
        OUTS = [256, 384, 512, 640, 768, 256]
        DROPS = [0.2]*3 + [0.3]*2
        
        self.model = nn.Sequential(*[oldJasperSubblock(n_mels, 256, 11, 2)] \
        + [JasperBlock(OUTS[i-1], OUTS[i], KERNELS[i], num_subblocks, DROPS[i])
           for i in range(5) for _ in range(num_blocks_factor)] \
        + [oldJasperSubblock(OUTS[-2], 896, 29, 1, 0.4, dilation=2)],
          oldJasperSubblock(896, 1024, 1, 1, 0.4), nn.Conv1d(1024, n_vocab, 1), nn.Softmax(1)
        )
        
    def forward(self, x):
        return self.model(x)