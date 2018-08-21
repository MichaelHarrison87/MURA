"""
Show the model.summary() of the various prebuilt CNN models available in keras.
The scripts for the models are located:
/home/harrisoneighty7_gmail_com/tensorflow/tensorflow/python/keras/applications
"""
from tensorflow.python.keras.applications import VGG16, VGG19, densenet, inception_resnet_v2, inception_v3
print("KERAS PREBUILT MODEL SUMMARIES")

# VGG16
vgg16 = VGG16()
print("VGG16:")
print(vgg16.summary())
print("~~~")
print()

# VGG16 - No Top
vgg16_notop = VGG16(include_top=False)
print("VGG16:")
print(vgg16_notop.summary())
print("~~~")
print()

# VGG19
vgg19 = VGG19()
print("VGG19:")
print(vgg19.summary())
print("~~~")
print()

# VGG19
vgg19_notop = VGG19(include_top=False)
print("VGG19:")
print(vgg19.summary())
print("~~~")
print()

## DenseNet
# # Code from densenet.py defines the various densenet architectures; each is made up of 4 blocks of various sizes:
#   # if blocks == [6, 12, 24, 16]:
#   #   model = Model(inputs, x, name='densenet121')
#   # elif blocks == [6, 12, 32, 32]:
#   #   model = Model(inputs, x, name='densenet169')
#   # elif blocks == [6, 12, 48, 32]:
#   #   model = Model(inputs, x, name='densenet201')
#   # else:
#   #   model = Model(inputs, x, name='densenet')
# densenet = densenet.DenseNet(blocks = [6, 12, 32, 32])
# print("DenseNet:")
# print(densenet.summary())
# print("~~~")
# print()
#
# # InceptionResNetV2
# inception_resnet_v2 = inception_resnet_v2.InceptionResNetV2()
# print("InceptionResNetV2 (Inception-v4):")
# print(inception_resnet_v2.summary())
# print("~~~")
# print()
#
# # Inception_V3
# inception_v3 = inception_v3.InceptionV3()
# print("Inception-v3:")
# print("~~~")
# print()
