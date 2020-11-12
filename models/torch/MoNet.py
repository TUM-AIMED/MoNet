import torch
import torch.nn as nn
import torch.nn.functional as F 

class ConvBnElu(nn.Module): 
    """
        Conv-Batchnorm-Elu block
    """
    def __init__(
        self,  
        old_filters,
        filters, 
        kernel_size=3, 
        strides=1, 
        dilation_rate=1
        ): 
        super(ConvBnElu, self).__init__()

        # Conv
        # 'SAME' padding => Output-Dim = Input-Dim/stride -> exact calculation: if uneven add more padding to the right
        # int() floors padding
        # TODO: how to add asymmetric padding? tuple option for padding only specifies the different dims 
        same_padding = int(dilation_rate*(kernel_size-1)*0.5)

        # TODO: kernel_initializer="he_uniform",

        self.conv = nn.Conv2d(
            in_channels=old_filters, 
            out_channels=filters, 
            kernel_size=kernel_size, 
            stride=strides, 
            padding=same_padding, 
            dilation=dilation_rate, 
            bias=False)

        # BatchNorm
        self.batch_norm = nn.BatchNorm2d(filters)

        #TODO: In paper there is a dropout layer at the end - left out in this implementation. (Should I include it?)

    def forward(self, x): 
        out = self.conv(x)
        out = self.batch_norm(out)
        out = F.elu(out)
        return out 

class deconv(nn.Module): 
    """
        Transposed Conv. with BatchNorm and ELU-activation
        Deconv upsampling of x. Doubles x and y dimension and maintains z.
    """
    def __init__(self, old_filters):
        super(deconv, self).__init__() 

        kernel_size = 4
        stride = 2
        dilation_rate = 1

        # TODO: how to add asymmetric padding? possibly use "output_padding here"
        same_padding = int(dilation_rate*(kernel_size-1)*0.5)

        # TODO: kernel_initializer="he_uniform",

        # TODO: here we conserve the number of channels, but in paper they are reduced to the half? 
        self.transp_conv = nn.ConvTranspose2d(
            in_channels=old_filters, 
            out_channels=old_filters, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=same_padding, 
            bias=False)

        self.batch_norm = nn.BatchNorm2d(old_filters)

    def forward(self, x):
        out = self.transp_conv(x)
        out = self.batch_norm(out)
        out = F.elu(out)
        return out 

class repeat_block(nn.Module): 
    """ 
        RDDC - Block
        Reccurent conv block with decreasing kernel size. 
        Makes use of atrous convolutions to make large kernel sizes computationally feasible

    """
    def __init__(
        self,  
        in_filters,
        out_filters, 
        dropout=0.2
        ): 
        super(repeat_block, self).__init__()

        # Skip connection 
        # TODO: Reformatting necessary?

        self.convBnElu1 = ConvBnElu(in_filters, out_filters, dilation_rate=4)
        self.dropout1 = nn.Dropout2d(dropout)
        self.convBnElu2 = ConvBnElu(out_filters, out_filters, dilation_rate=3)
        self.dropout2 = nn.Dropout2d(dropout)
        self.convBnElu3 = ConvBnElu(out_filters, out_filters, dilation_rate=2)
        self.dropout3 = nn.Dropout2d(dropout)
        self.convBnElu4 = ConvBnElu(out_filters, out_filters, dilation_rate=1)

    def forward(self, x): 
        skip1 = x
        out = self.convBnElu1(x)
        out = self.dropout1(out)
        out = self.convBnElu2(out + skip1)
        out = self.dropout2(out)
        skip2 = out
        out = self.convBnElu3(out)
        out = self.dropout3(out)
        out = self.convBnElu4(out + skip2)

        #TODO: In this implementation there was again a skip connection from first input, not shown in paper however? 
        out = skip1 + out
        return out


class MoNet(nn.Module): 
    def __init__(
        self, 
        input_shape=(256, 256, 1),
        output_classes=1,
        depth=2,
        n_filters_init=16,
        dropout_enc=0.2,
        dropout_dec=0.2,
        ):
        super(MoNet, self).__init__()
        
        # store param in case they're needed later
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.depth = depth
        self.features = n_filters_init
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec
        
        # encoder
        encoder_list = []

        old_filters = 1
        features = n_filters_init
        for i in range(depth):
            encoder_list.append([
                f"Enc_ConvBnElu_Before_{i}", ConvBnElu(old_filters, features)
                ])
            old_filters = features
            encoder_list.append([
                f"Enc_RDDC_{i}", repeat_block(old_filters, features, dropout=dropout_enc)
                ])
            encoder_list.append([
                f"Enc_ConvBnElu_After_{i}", ConvBnElu(old_filters, features, kernel_size=4, strides=2)
                ])
            features *= 2

        # ModulList instead of Sequential because we don't want the layers to be connected yet 
        # we still need to add the skip connections. Dict to know when to add skip connection in forward
        self.encoder = nn.ModuleDict(encoder_list)

        # bottleneck
        bottleneck_list = []
        bottleneck_list.append(ConvBnElu(old_filters, features))
        old_filters = features
        bottleneck_list.append(repeat_block(old_filters, features))

        self.bottleneck = nn.Sequential(*bottleneck_list)

        # decoder
        decoder_list = []
        for i in reversed(range(depth)):
            features //= 2
            decoder_list.append([
                f"Dec_deconv_Before_{i}", deconv(old_filters)
                ])
            # deconv maintains number of channels 
            decoder_list.append([
                f"Dec_ConvBnElu_{i}", ConvBnElu(old_filters, features)
                ])
            old_filters = features
            decoder_list.append([
                f"Dec_RDDC_{i}", repeat_block(old_filters, features, dropout=dropout_dec)
                ])

        self.decoder = nn.ModuleDict(decoder_list)

        # head
        head_list = []
        # TODO: kernel_initializer="he_uniform",
        head_list.append(nn.Conv2d(
            in_channels=old_filters, 
            out_channels=output_classes, 
            kernel_size=1, 
            stride=1, 
            bias=False))

        head_list.append(nn.BatchNorm2d(output_classes))

        # TODO: Consider nn.logsoftmax --> works with NLLoss out of the box --> what we want to use. 
        if output_classes > 1:
            activation = nn.Softmax(dim=1)
        else:
            activation = nn.Sigmoid()
        head_list.append(activation)

        self.header = nn.Sequential(*head_list)

    def forward(self, x): 
        skip = []

        # encoder 
        out = x
        for key in self.encoder: 
            out = self.encoder[key](out)
            if key == "RDDC": 
                skip.append(out)

        # bottleneck 
        out = self.bottleneck(out)

        # decoder 
        for key in self.decoder: 
            out = self.decoder[key](out)
            if key == "deconv": 
                # Concatenate along channel-dim (last dim)
                # skip.pop() -> get last element and remove it 
                out = torch.cat((out, skip.pop()), dim=-1)

        # header
        out = self.header(out)

        return out