# ACDC 2017
MICCAI challenge for ACDC 2017

# Instruction for Tensorboardx

pip install tensorboardX
pip install tensorflow

start tensorboard by "tensorboard --logdir=<dir_to_store_log_file>"

# Logs
1. vanilla_trained_unet_limited_data: 0.8430

2. aug_trained_unet: 0.8465 (with fine tuning)

3. UNet training performance (unfirom weight):
EPOCH 70 of 70

Training Loss: 2.4717
0 Class, True Pos 57622332.0, False Pos 230931.0, Flase Neg 135884.0, Dice score 1.0
1 Class, True Pos 481663.0, False Pos 87809.0, Flase Neg 96207.0, Dice score 0.84
2 Class, True Pos 483941.0, False Pos 156754.0, Flase Neg 167215.0, Dice score 0.75
3 Class, True Pos 541580.0, False Pos 32748.0, Flase Neg 108936.0, Dice score 0.88

4. UNet training performance (class balanced weight):
EPOCH 70 of 70

Training Loss: 3.5021
0 Class, True Pos 57608688.0, False Pos 289468.0, False Neg 149521.0, Dice score 1.00
1 Class, True Pos 460065.0, False Pos 90767.0, False Neg 117805.0, Dice score 0.82
2 Class, True Pos 465871.0, False Pos 154939.0, False Neg 185285.0, Dice score 0.73
3 Class, True Pos 533709.0, False Pos 34244.0, False Neg 116807.0, Dice score 0.88

5. UNet training performance (weight: inv(10 2 1 2)): 0.8678
EPOCH 70 of 70

Training Loss: 2.0016
0 Class, True Pos 57672216.0, False Pos 151085.0, False Neg 78722.0, Dice score 1.00
1 Class, True Pos 504550.0, False Pos 46985.0, False Neg 86626.0, Dice score 0.88
2 Class, True Pos 553362.0, False Pos 90541.0, False Neg 92723.0, Dice score 0.86
3 Class, True Pos 596317.0, False Pos 22704.0, False Neg 53244.0, Dice score 0.94

6. UNet+ResNet training performance (weight: inv(10 2 1 2) + 0.95 weight decay/epoch + 150 epochs): 0.8901
EPOCH 150 of 150

Training Loss: 0.8127
0 Class, True Pos 57693812.0, False Pos 40665.0, False Neg 37042.0, Dice score 1.00
1 Class, True Pos 566957.0, False Pos 24066.0, False Neg 27550.0, Dice score 0.96
2 Class, True Pos 623537.0, False Pos 40679.0, False Neg 34749.0, Dice score 0.94
3 Class, True Pos 635007.0, False Pos 13037.0, False Neg 19106.0, Dice score 0.98

7. UNet+ResNet training performance (prev + augmentation):
