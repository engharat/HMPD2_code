import logging

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from CNN_model import CNNModel
from spectral_former import ViT
from Rpnet import IncUNet
from hvt import SSRN_network
from Vision_transformer import ViT_2
from HybridSN import HybridSN
#from MobileNetv2 import MobileNetV2, get_fine_tuning_parameters, get_model
import torch
from  torchvision.models import resnet50, ResNet50_Weights
from swin_transformer_pytorch import SwinTransformer
from torch import nn
from torchvision.models import swin_t
from MobileNet_v2_3D import MobileNetV2
#from Swin3D import Swin3DUNet
from transformers import TimesformerConfig, TimesformerModel, TimesformerForVideoClassification
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from pytorchvideo.models import create_resnet
import torchvision.models.video as video_models
from transformers import VivitImageProcessor, VivitModel, VivitConfig
from torchvision.models.video import swin3d_t
class AddConstant(nn.Module):
    def __init__(self, constant):
        super(AddConstant, self).__init__()
        self.constant = constant

    def forward(self, x):
        return x + self.constant

def getTransformer(transform_resize, transform_crop, transform_normalize_mean, transform_normalize_var):

    transform = transforms.Compose(
            [
                transforms.Resize(transform_resize),
                transforms.RandomCrop(transform_crop),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(transform_normalize_mean, transform_normalize_var),
            ]
        )

    return transform

## aggiungere rete Spectralformer

def generateModel(desired_model, num_classes):
    if desired_model == 'alexNet':
        model = models.alexnet(weights='IMAGENET1K_V1')
        model.classifier[6] = nn.Linear(4096, num_classes)

    if desired_model == 'resNet18':
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    if desired_model == 'resnext50_32x4d':
        model = models.resnext50_32x4d(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        

    if desired_model == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        #import pdb; pdb.set_trace()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    if desired_model == 'swin':
        model = models.swin_v2_b(weights='IMAGENET1K_V1')
        #import pdb; pdb.set_trace()
        num_ftrs = model.head.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)


    if desired_model == 'vit_b_16':
        model = models.vit_b_16(weights='IMAGENET1K_V1')
#        import pdb; pdb.set_trace()
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)

    if desired_model == 'vgg11':
        model = models.vgg11()

    if desired_model == 'mobilenet_v2':
        model = models.mobilenet_v2()

    if desired_model == 'CNNModel':
    
        model = CNNModel(input_channels=24)
        
    if desired_model == 'SpectralFormer':
        model =  ViT(
    image_size = 80,
    near_band = 1,
    num_patches = 24,
    num_classes = 2,
    dim = 80*80,
    depth = 5,
    heads = 4,
    mlp_dim = 8,
    dropout = 0.1,
    emb_dropout = 0.1,
    
)
    if desired_model == 'RpNet':
    
        model = IncUNet(in_shape = (24,80,80))
        
    if desired_model == 'HVT':
    
        model = SSRN_network(24, 2)
        
    if desired_model == 'HVT_transformer':
    
        model = ViT_2(
    image_size = 80,
    patch_size = 1,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    channels = 24,
    dropout = 0.1,
    emb_dropout = 0.1
)
    if desired_model == 'HybridSN':
    
        model = HybridSN(24,80,2)
        
    if desired_model == 'MobileNetv2':
    
        # Load a pre-trained ResNet model
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')

        # Modify the first convolutional layer to accept 24 channels instead of 3
        model.features[0][0] = nn.Conv2d(24, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Get the number of input features for the last layer
        #num_ftrs = model.fc.in_features

        # Replace the last fully connected layer with a new one (e.g., for 10 classes)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        
    if desired_model == 'ResNet':
    
        # Load a pre-trained ResNet model
        model = models.resnet18(weights='IMAGENET1K_V1')

        # Modify the first convolutional layer to accept 24 channels instead of 3
        model.conv1 = nn.Conv2d(24, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Get the number of input features for the last layer
        num_ftrs = model.fc.in_features

        # Replace the last fully connected layer with a new one (e.g., for 10 classes)
        model.fc = nn.Linear(num_ftrs, 2)
        
    if desired_model == 'ResNet50':
    
        # Load a pre-trained ResNet model
        model = models.resnet50(weights='IMAGENET1K_V1')

        # Modify the first convolutional layer to accept 24 channels instead of 3
        model.conv1 = nn.Conv2d(24, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Get the number of input features for the last layer
        num_ftrs = model.fc.in_features

        # Replace the last fully connected layer with a new one (e.g., for 10 classes)
        model.fc = nn.Linear(num_ftrs, 2)    
        
        
    if desired_model == 'Vit_b16':
    
       # Load a pre-trained ResNet model
       model = models.vit_b_16(weights='IMAGENET1K_V1')

       # Modify the input embedding layer to accept 24 input channels
       original_embedding = model.conv_proj
       new_embedding = nn.Conv2d(24, original_embedding.out_channels, kernel_size=original_embedding.kernel_size, 
                          stride=original_embedding.stride, padding=original_embedding.padding, bias=original_embedding.bias is not None)

       # Copy the pretrained weights for the first 3 channels
       with torch.no_grad():
         new_embedding.weight[:, :3, :, :] = original_embedding.weight
         if new_embedding.weight.size(1) > 3:
            new_embedding.weight[:, 3:, :, :].zero_()

       # Replace the embedding layer in the model
       model.conv_proj = new_embedding

       # Adjust the final classification layer to output 2 classes
       num_features = model.heads[-1].in_features
       model.heads[-1] = nn.Linear(num_features, 2)
       
    if desired_model == 'Swin_v2':
    
       # Load a pre-trained ResNet model
       model = models.swin_v2_b(weights='IMAGENET1K_V1')

       # Modify the input embedding layer to accept 24 input channels
       original_conv = model.features[0][0]
       new_conv = nn.Conv2d(24, original_conv.out_channels, kernel_size=original_conv.kernel_size, 
                     stride=original_conv.stride, padding=original_conv.padding, bias=original_conv.bias is not None)

        # Copy the pretrained weights for the first 3 channels
       with torch.no_grad():
           new_conv.weight[:, :3, :, :] = original_conv.weight
           if new_conv.weight.size(1) > 3:
              new_conv.weight[:, 3:, :, :].zero_()

       # Replace the first convolutional layer
       model.features[0][0] = new_conv
       num_features = model.head.in_features
       model.head = nn.Linear(num_features, 2) 
    if desired_model == 'Vit_b16_2channel':
    
       # Load a pre-trained Vision Transformer model
       model = models.vit_b_16(weights='IMAGENET1K_V1')

       # Modify the input embedding layer to accept 2 input channels
       original_embedding = model.conv_proj
       new_embedding = nn.Conv2d(2, original_embedding.out_channels, kernel_size=original_embedding.kernel_size, 
                          stride=original_embedding.stride, padding=original_embedding.padding, bias=original_embedding.bias is not None)

       # Copy the pretrained weights for the first 2 channels
       with torch.no_grad():
            new_embedding.weight[:, :2, :, :] = original_embedding.weight[:, :2, :, :]
            if new_embedding.weight.size(1) > 2:
               new_embedding.weight[:, 2:, :, :].zero_()

       # Replace the embedding layer in the model
       model.conv_proj = new_embedding

       # Adjust the final classification layer to output 2 classes
       num_features = model.heads.head.in_features
       model.heads.head = nn.Linear(num_features, 2) 
       
    if desired_model == 'Vit_b16_3channel':
    
       # Load a pre-trained Vision Transformer model
       model = models.vit_b_16(weights='IMAGENET1K_V1')

       
       # Adjust the final classification layer to output 2 classes
       num_features = model.heads.head.in_features
       model.heads.head = nn.Linear(num_features, 2)    

    if desired_model == 'Swin_compressed':
    
        model = swin_t(weights='IMAGENET1K_V1')
        original_conv = model.features[0][0]
        compression_conv = nn.Conv2d(24, 3, kernel_size=3, 
                     stride=1, padding=1, bias=True)
        relu_layer = nn.ReLU()
        add_layer = AddConstant(-0.5)
        seq_layer = nn.Sequential(compression_conv,
                                  relu_layer,
                                  add_layer,
                                  original_conv
                                  )
      

        

       # Replace the first convolutional layer
        model.features[0][0] = seq_layer
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, 2)
        
    if desired_model == 'Swin_no_relu':
    
        model = swin_t(weights='IMAGENET1K_V1')
        original_conv = model.features[0][0]
        compression_conv = nn.Conv2d(24, 3, kernel_size=3, 
                     stride=1, padding=1, bias=True)
        relu_layer = nn.ReLU()
        add_layer = AddConstant(-0.5)
        seq_layer = nn.Sequential(compression_conv,
                                  original_conv
                                  )
      

        

       # Replace the first convolutional layer
        model.features[0][0] = seq_layer
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, 2)    
    if desired_model == 'Swin_normal':
    
        model = swin_t(weights='IMAGENET1K_V1')
        original_conv = model.features[0][0]
        new_conv = nn.Conv2d(24, original_conv.out_channels, kernel_size=original_conv.kernel_size, 
                     stride=original_conv.stride, padding=original_conv.padding, bias=original_conv.bias is not None)

        # Copy the pretrained weights for the first 3 channels
        with torch.no_grad():
           new_conv.weight[:, :3, :, :] = original_conv.weight
           if new_conv.weight.size(1) > 3:
              new_conv.weight[:, 3:, :, :].zero_()

       # Replace the first convolutional layer
        model.features[0][0] = new_conv
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, 2)    
        
    if desired_model == 'Swin_no_copy':
    
        model = swin_t(weights='IMAGENET1K_V1')
        original_conv = model.features[0][0]
        new_conv = nn.Conv2d(24, original_conv.out_channels, kernel_size=original_conv.kernel_size, 
                     stride=original_conv.stride, padding=original_conv.padding, bias=original_conv.bias is not None)

        # Copy the pretrained weights for the first 3 channels
        #with torch.no_grad():
           #new_conv.weight[:, :3, :, :] = original_conv.weight
           #if new_conv.weight.size(1) > 3:
           #   new_conv.weight[:, 3:, :, :].zero_()

       # Replace the first convolutional layer
        model.features[0][0] = new_conv
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, 2)   
    if desired_model =="MobileNet_v2_3D":
       model = MobileNetV2(num_classes = 2)
       model.load_state_dict(torch.load('kinetics_mobilenetv2_1.0x_RGB_16_best.pth'),strict = False)        
        
    if desired_model =="Swin3D":
        model = Swin3DUNet(8, 24, 2, \
        224, 8, up_k=up_k, \
        drop_path_rate=drop_path_rate, num_classes=2, \
        num_layers=num_layers, stem_transformer=stem_transformer, \
        upsample=upsample, first_down_stride=down_stride, \
        knn_down=knn_down, in_channels=in_channels, \
        cRSE='XYZ_RGB_NORM', fp16_mode=1)
        model.load_pretrained_model("Swin3D_RGBN_S.pth")
    if desired_model == "TimeSformer":
        configuration = TimesformerConfig(num_frames = 8)
        model = TimesformerForVideoClassification(configuration).from_pretrained("facebook/timesformer-base-finetuned-k400")
        model.config.return_dict=False
        model.config.num_frames=8
        model.classifier = nn.Linear(768, 2)
        #import pdb; pdb.set_trace()
    if desired_model == "Resnet_3D":
        model = video_models.r3d_18(weights='R3D_18_Weights.KINETICS400_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    if desired_model == "VideoMae":
       model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics") 
       model.config.return_dict = False
       model.config.num_frames = 24
       #model.config.patch_size = 4
       #import pdb; pdb.set_trace()
       model.classifier = nn.Linear(768, 2)
    if desired_model == "Vivit" :
       configuration =  VivitConfig(num_frames = 24)
       model = VivitModel(configuration).from_pretrained("google/vivit-b-16x2-kinetics400") 
       model.config.return_dict = False
       model.config.num_frames = 24
       model.classifier = nn.Linear(768, 2)
    if desired_model == "Mvit":
        weights = video_models.MViT_V1_B_Weights.KINETICS400_V1
        model = video_models.mvit_v1_b(weights=weights)
        num_features = model.head[-1].in_features
        model.head[-1] = torch.nn.Linear(num_features, 2)
    if desired_model == "S3D":
        weights = video_models.S3D_Weights.DEFAULT
        model = video_models.s3d(weights=weights)  
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, 2)
    if desired_model == "VideoSwin":
       model = swin3d_t(weights='KINETICS400_V1')
       model.head = nn.Linear(model.head.in_features, 2)   
        
    if "model" in locals():
        return model
    else:
        logging.error(f'the name of the network {desired_model} is not in the available models list')
        
