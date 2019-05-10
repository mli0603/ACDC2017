# ACDC 2017
MICCAI challenge for ACDC 2017

# Instruction for Tensorboardx

pip install tensorboardX
pip install tensorflow

start tensorboard by "tensorboard --logdir=<dir_to_store_log_file>"

# Logs
1. vanilla_trained_unet_limited_data: 0.8430

2. aug_trained_unet: 0.8465 (with fine tuning)

3. vanilla unet training performance (unfirom weight):
EPOCH 70 of 70

Training Loss: 2.4717
0 Class, True Pos 57622332.0, False Pos 230931.0, Flase Neg 135884.0, Dice score 1.0
1 Class, True Pos 481663.0, False Pos 87809.0, Flase Neg 96207.0, Dice score 0.84
2 Class, True Pos 483941.0, False Pos 156754.0, Flase Neg 167215.0, Dice score 0.75
3 Class, True Pos 541580.0, False Pos 32748.0, Flase Neg 108936.0, Dice score 0.88

4. vanilla unet training performance (class balanced weight):
EPOCH 70 of 70

Training Loss: 3.5021
0 Class, True Pos 57608688.0, False Pos 289468.0, False Neg 149521.0, Dice score 1.00
1 Class, True Pos 460065.0, False Pos 90767.0, False Neg 117805.0, Dice score 0.82
2 Class, True Pos 465871.0, False Pos 154939.0, False Neg 185285.0, Dice score 0.73
3 Class, True Pos 533709.0, False Pos 34244.0, False Neg 116807.0, Dice score 0.88

5. vanilla unet training performance (weight: inv(10 2 1 2)):
EPOCH 70 of 70

Training Loss: 2.9033
0 Class, True Pos 57636256.0, False Pos 212274.0, False Neg 121962.0, Dice score 1.00
1 Class, True Pos 472704.0, False Pos 76667.0, False Neg 105166.0, Dice score 0.84
2 Class, True Pos 518519.0, False Pos 134326.0, False Neg 132637.0, Dice score 0.80
3 Class, True Pos 558236.0, False Pos 28778.0, False Neg 92280.0, Dice score 0.90
