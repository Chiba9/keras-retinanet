"""
Copyright 2017-2018 cgratie (https://github.com/cgratie/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import keras
from keras.utils import get_file

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image

class EfficientNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return efficientnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
        """
        model_list = ['efficientnet-b0',
                      'efficientnet-b1',
                      'efficientnet-b2',
                      'efficientnet-b3',
                      'efficientnet-b4',
                      'efficientnet-b5',
                      'efficientnet-b6',
                      'efficientnet-b7']
        if self.backbone in model_list:
            resource = 'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/'+self.backbone+'_notop.h5'
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))

        return get_file(
            '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
            resource,
            cache_subdir='models',
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['efficientnet-b0',
                      'efficientnet-b1',
                      'efficientnet-b2',
                      'efficientnet-b3',
                      'efficientnet-b4',
                      'efficientnet-b5',
                      'efficientnet-b6',
                      'efficientnet-b7']
        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')

from keras_efficientnets import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

def efficientnet_retinanet(num_classes, backbone='efficientnet-b0', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a vgg backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the vgg backbone
    if backbone == 'efficientnet-b0':
        efficientnet = EfficientNetB0(input_tensor = inputs, include_top=False, weights='imagenet')
        layer_names = ["swish_16", "swish_34", "swish_49"]
    elif backbone == 'efficientnet-b1':
        efficientnet = EfficientNetB1(input_tensor = inputs, include_top=False, weights='imagenet')
        layer_names = ["swish_24", "swish_48", "swish_69"]
    elif backbone == 'efficientnet-b2':
        efficientnet = EfficientNetB2(input_tensor = inputs, include_top=False, weights='imagenet')
        layer_names = ["swish_24", "swish_48", "swish_69"] 
    elif backbone == 'efficientnet-b3':
        efficientnet = EfficientNetB3(input_tensor = inputs, include_top=False, weights='imagenet')
        layer_names = ["swish_24", "swish_54", "swish_78"]
    elif backbone == 'efficientnet-b4':
        efficientnet = EfficientNetB4(input_tensor = inputs, include_top=False, weights='imagenet')
        layer_names = ["swish_30", "swish_66", "swish_96"]
    elif backbone == 'efficientnet-b':
        efficientnet = EfficientNetB4(input_tensor = inputs, include_top=False, weights='imagenet')
        layer_names = ["swish_30", "swish_66", "swish_96"]
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))
        

    if modifier:
        efficientnet = modifier(efficientnet)


    efficientnet.summary()
    # create the full model
    
    layer_outputs = [efficientnet.get_layer(name).output for name in layer_names]
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
