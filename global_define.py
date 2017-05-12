#########################LAYER1##################################
NUM_CLASSES = 20

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 224
IMAGE_PIXELS = 224*224
INPUT_SIZE=224
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE='test.tfrecords' 
#IMAGE_PIXELS=224*224
momentum=0.5
epsilonw=0.05
epsilonb=0.05
epsilona=0.05
weightcost=0.00002

BATCH_SIZE=256
#############LAYER3########################################
# NUM_CLASSES = 20

# # The MNIST images are always 28x28 pixels.
# IMAGE_SIZE = 224
# IMAGE_PIXELS = 224*224

# momentum=0.5
# epsilonw=0.1#0.05
# epsilonb=0.1#0.05
# epsilona=0.1#0.05
# weightcost=0.0002

# BATCH_SIZE=100
