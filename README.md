## Data AUDIT: Identifying Attribute Utility- and Detectability-Induced Bias in Task Models
### MICCAI 2023
We seek to audit at the dataset level to develop targeted hypotheses on the bias downstream models will inherit. We focus on identifying potential shortcuts, and define two metrics we term “utility” and “detectability” respectively. Utility measures how much information knowing an attribute conveys about the task label. For detectability, we seek to measure how well a downstream model could extract the values of the attribute from the image set, excluding task related information. 
ArXiv paper link: https://arxiv.org/abs/2304.03218
(MICCAI 2023 link pending): 
Code Information:
Please note, full results require training of several hundred models. For your convenience, the codebase is split into separate config files to recreate each experiment. 
