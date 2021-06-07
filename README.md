# Dialogue_System_Hackathon

This repository consists of code, processed data and trained models leveraged in [DiSCoL: Toward Engaging Dialogue Systems through Conversational
Line Guided Response Generation](https://www.aclweb.org/anthology/2021.naacl-demos.4.pdf). Please cite this work as: @inproceedings{ghazarian2021discol, title={DiSCoL: Toward Engaging Dialogue Systems through Conversational Line Guided Response Generation}, author={Sarik Ghazarian and Zixi Liu and Tuhin Chakrabarty and Xuezhe Ma and Aram Galstyan and Nanyun Peng}, booktitle={2021 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), Demonstrations Track}, pages={26–34}, year={2021} }


Please feel free to contact [me](mailto:sarikgha@usc.edu) for suggestions or any issues. 

## Steps to run the demo

### Environment Creation
Create a new environment from the webdemo.yml file
webdemo.yml file includes all the necessary libraries and packages to run the demo.

## Load the models
We have four pretrained models that should be loaded to run the demo:
1. ent_kwd: which predicts keywords given the dialogue context.
2. topic_cls: which classifies each utterance in one of the existing topics.
3. dialogpt: which is finetuned version of dialogpt on our data. It generates next utterance (response) given dialogue context, predicted keywords and topic.
4. baseline: is the pretrained dialogpt model that we consider as the baseline model to generate response (it doesn't include the predicted keywords).
All these models have been uploaded: 

Download all these models and put them in a folder like: ./Models/
Then try to update all the paths in the first four lines of webdemo/SETTING.py file accordingly.


## Access to a remote machine
Try to run the demo on a machine with GPUs (You can remotely access the machine using coomand: ssh -L PORT_NUMBER:127.0.0.1:PORT_NUMBER SERVER_NAME)

## Run demo on GPU 
CUDA_VISIBLE_DEVICES=0 python webdemo/app.py

## Converse with DiSCoL! 
In your local browser, try to connect to the server: http://127.0.0.1:PORT_NUMBER

Enjoy conversing with DiSCoL!





