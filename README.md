# COMP 551, Third Project, Group 27

Members: Dailun Li, Kevin Xu, Zijun Zhao

Code for COMP 551 comptetition ''Multi-label classification of handwritten digits and english alphabet''

Competition link: https://www.kaggle.com/c/comp-551-fall-2021

The model is similar to AlexNet. Preprocessing techniques applied include augmentation by rotation, scaling up by a factor of 2, and applying one-hot encoding to labels. FixMatch is used to utilize the unlabeled data. The final prediction is calculated from the weighted average of different versions of model, with weights chosen from experiments on the validation set.
