DIALOGPT = '/nas/home/zixiliu/dialogsystem_model/bartgen/'
CONV_LINE = '/nas/home/zixiliu/dialogsystem_model/ent_kwd/'
TOPIC = '/nas/home/zixiliu/dialogsystem_model/topic_cls/'
parser = {}
parser['seed'] = 1000
parser['gen_model_type'] = "bart"
parser['gen_model_path'] = DIALOGPT
parser['conv_line_path'] = CONV_LINE
parser['topic_cls_path'] = TOPIC
parser['label_dir'] = TOPIC
parser['length'] = 100
parser['temperature'] = 1.0
parser['repetition_penalty'] = 1.0
parser['top_k'] = 0
parser['top_p'] = 0.9
parser['stop_token'] = '\n'