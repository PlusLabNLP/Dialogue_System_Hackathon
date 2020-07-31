# Dialogue_System_Hackathon

#1. Preprocessing Topical chat dataset:
This is step includes extracting entities, keywords from utterances and assigning topic to utterances based on the its provided knowledge_source.

python preprocess.py --fname=valid_freq --data_dir=alexa-prize-topical-chat-dataset/ --mode=extract
