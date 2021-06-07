# Dialogue_System_Hackathon

This repository consists of code, processed data and trained models leveraged in [DiSCoL: Toward Engaging Dialogue Systems through Conversational
Line Guided Response Generation](https://www.aclweb.org/anthology/2021.naacl-demos.4.pdf). Please cite this work as: @inproceedings{ghazarian2021discol, title={DiSCoL: Toward Engaging Dialogue Systems through Conversational Line Guided Response Generation}, author={Sarik Ghazarian and Zixi Liu and Tuhin Chakrabarty and Xuezhe Ma and Aram Galstyan and Nanyun Peng}, booktitle={2021 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), Demonstrations Track}, pages={26–34}, year={2021} }


Please feel free to contact [me](mailto:sarikgha@usc.edu) for any suggestions or issues. 

## Steps to run the demo

### Environment Creation
Create a new environment from the webdemo.yml file which includes all the necessary libraries and packages to run the demo.

## Load the models
We have four models that should be loaded to run the DiSCoL:
1. ent_kwd: is a finetuned BART (Lewis et al., 2019) model that predicts convlines given the dialogue context utterance, entities and topics.
2. topic_cls: is a finetuned BERT (Devlin et al., 2019) that predicts a topic label for each given dialogue utterance.
3. bartgen: is a finetuned  BART (Lewis et al., 2019) model that generates next utterance (response) given dialogue context utterance, predicted convlines and topic.
4. baseline: is the pretrained DialoGPT (Zhang et al., 2019) model that we consider as the baseline model to generate response (it doesn't take the predicted keywords and topic as the input to generate the response).
All these models can be downloaded from [here] (https://drive.google.com/drive/folders/15ML4UyaCko4e7qxOyP0WRbMidWSph36b?usp=sharing)

Download all these models and put them in a folder (eq. ./Models).
Then try to update all the paths in the first four lines of webdemo/SETTING.py file accordingly such that DiSCoL would be able to locate and load them correctly.


## Access to a remote machine
Try to run the demo on a machine with GPUs (You can remotely access the machine using coomand: ssh -L PORT_NUMBER:127.0.0.1:PORT_NUMBER SERVER_NAME)

## Run demo 
It is encouraged to run DiSCoL on a machine with GPUs. If your machine does not have a GPU, access to a machine with GPUs remotely using ssh -L PORT_NUMBER:127.0.0.1:PORT_NUMBER MACHINE_NAME.

On the connected server run the DiSCoL on a GPU: python webdemo/app.py

## Converse with DiSCoL! 
In your local browser, try to connect to the server: http://127.0.0.1:PORT_NUMBER

The DiCoL should be ready to converse. Enjoy conversing with DiSCoL!





