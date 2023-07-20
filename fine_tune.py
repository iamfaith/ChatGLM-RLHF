#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# 导入常用模块
import numpy as np
import pandas as pd 
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader 
# 配置参数
from argparse import Namespace

cfg = Namespace()

#dataset
cfg.prompt_column = 'prompt'
cfg.response_column = 'response'
cfg.history_column = None
cfg.source_prefix = '' #添加到每个prompt开头的前缀引导语

cfg.max_source_length = 128 
cfg.max_target_length = 128

#model
cfg.model_name_or_path = "/home/faith/chatglm2-6b"  #远程'THUDM/chatglm-6b' 
cfg.quantization_bit = None #仅仅预测时可以选 4 or 8 


#train
cfg.epochs = 100 
cfg.lr = 5e-3
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 16 #梯度累积


import transformers
from transformers import  AutoModel,AutoTokenizer,AutoConfig,DataCollatorForSeq2Seq


config = AutoConfig.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    cfg.model_name_or_path, trust_remote_code=True)

model = AutoModel.from_pretrained(cfg.model_name_or_path,config=config,
                                  trust_remote_code=True).half() 
#先量化瘦身
if cfg.quantization_bit is not None:
    print(f"Quantized to {cfg.quantization_bit} bit")
    model = model.quantize(cfg.quantization_bit)
    
#再移动到GPU上
model = model.cuda()





#%% 
import datasets 

#定义一条知识样本~

keyword = '梦中情炉'

description = '''梦中情炉一般指的是炼丹工具torchkeras。
这是一个通用的pytorch模型训练模版工具。
torchkeras是一个三好炼丹炉：好看，好用，好改。
她有torch的灵动，也有keras的优雅，并且她的美丽，无与伦比。
所以她的作者一个有毅力的吃货给她取了一个别名叫做梦中情炉。'''

#对prompt使用一些简单的数据增强的方法，以便更好地收敛。
def get_prompt_list(keyword):
    return [f'{keyword}', 
            f'你知道{keyword}吗?',
            f'{keyword}是什么？',
            f'介绍一下{keyword}',
            f'你听过{keyword}吗?',
            f'啥是{keyword}？',
            f'{keyword}是何物？',
            f'何为{keyword}？',
           ]

data =[{'prompt':x,'response':description} for x in get_prompt_list(keyword) ]
dfdata = pd.DataFrame(data)

#训练集和验证集一样
ds_train_raw = ds_val_raw = datasets.Dataset.from_pandas(dfdata)

#%% data transformation
#这是支持 history列处理，并且按照batch预处理数据的方法。

def preprocess(examples):
    max_seq_length = cfg.max_source_length + cfg.max_target_length
    model_inputs = {
        "input_ids": [],
        "labels": [],
    }
    for i in range(len(examples[cfg.prompt_column])):
        if examples[cfg.prompt_column][i] and examples[cfg.response_column][i]:
            query, answer = examples[cfg.prompt_column][i], examples[cfg.response_column][i]

            history = examples[cfg.history_column][i] if cfg.history_column is not None else None
            prompt = tokenizer.build_prompt(query, history)

            prompt = cfg.source_prefix + prompt
            a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                     max_length=cfg.max_source_length)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                     max_length=cfg.max_target_length)

            context_length = len(a_ids)
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
    return model_inputs

ds_train = ds_train_raw.map(
    preprocess,
    batched=True,
    num_proc=4,
    remove_columns=ds_train_raw.column_names
)

ds_val = ds_val_raw.map(
    preprocess,
    batched=True,
    num_proc=4,
    remove_columns=ds_val_raw.column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=None,
    label_pad_token_id=-100,
    pad_to_multiple_of=None,
    padding=False
)

dl_train = DataLoader(ds_train,batch_size = cfg.batch_size,
                      num_workers = 2, shuffle = True, collate_fn = data_collator 
                     )
dl_val = DataLoader(ds_val,batch_size = cfg.batch_size,
                      num_workers = 2, shuffle = False, collate_fn = data_collator 
                     )


#%%
from peft import get_peft_model, AdaLoraConfig, TaskType

#训练时节约GPU占用
model.config.use_cache=False
model.supports_gradient_checkpointing = True  #
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

peft_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1,
    target_modules=["query", "value"]
)

peft_model = get_peft_model(model, peft_config)

peft_model.is_parallelizable = True
peft_model.model_parallel = True
peft_model.print_trainable_parameters()

for name,para in peft_model.named_parameters():
    if '.2.' in name:
        break 
    if 'lora' in name.lower():
        print(name+':')
        print('shape = ',list(para.shape),'\t','sum = ',para.sum().item())
        print('\n')


# %% train model

from torchkeras import KerasModel 
from accelerate import Accelerator 

class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator() 
        if self.stage=='train':
            self.net.train() 
        else:
            self.net.eval()
    
    def __call__(self, batch):
        
        #loss
        with self.accelerator.autocast():
            loss = self.net(input_ids=batch["input_ids"],labels=batch["labels"]).loss

        #backward()
        if self.optimizer is not None and self.stage=="train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
        all_loss = self.accelerator.gather(loss).sum()
        
        #losses (or plain metrics that can be averaged)
        step_losses = {self.stage+"_loss":all_loss.item()}
        
        #metrics (stateful metrics)
        step_metrics = {}
        
        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics
    
KerasModel.StepRunner = StepRunner 


#仅仅保存lora可训练参数
def save_ckpt(self, ckpt_path='checkpoint', accelerator = None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)
    
def load_ckpt(self, ckpt_path='checkpoint'):
    import os
    self.net.load_state_dict(
        torch.load(os.path.join(ckpt_path,'adapter_model.bin')),strict =False)
    self.from_scratch = False
    
KerasModel.save_ckpt = save_ckpt 
KerasModel.load_ckpt = load_ckpt 



optimizer = torch.optim.AdamW(peft_model.parameters(),lr=cfg.lr) 
keras_model = KerasModel(peft_model,loss_fn = None,
        optimizer=optimizer) 
ckpt_path = 'single_chatglm2'

keras_model.fit(train_data = dl_train,
                val_data = dl_val,
                epochs=100,
                patience=20,
                monitor='val_loss',
                mode='min',
                ckpt_path = ckpt_path,
                # mixed_precision='no',
                mixed_precision='fp16',
                gradient_accumulation_steps = cfg.gradient_accumulation_steps
               )

# print("----", type(self.metrics_dict), self.accelerator.unwrap_model(model=self.metrics_dict))        


# %%
# 通过注册jupyter魔法命令可以很方便地在jupyter中测试ChatGLM 
from torchkeras.chat import ChatGLM 
chatglm = ChatGLM(peft_model,tokenizer)

#%%

# pip install protobuf==3.20.1



# %% validate
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.system('export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python')
from torchkeras.chat import ChatGLM 
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel 
ckpt_path = 'single_chatglm2'
model_old = AutoModel.from_pretrained("/home/faith/chatglm2-6b",
                                  load_in_8bit=False, 
                                  trust_remote_code=True)
peft_loaded = PeftModel.from_pretrained(model_old, ckpt_path).cuda()
model_new = peft_loaded.merge_and_unload() #合并lora权重

tokenizer = AutoTokenizer.from_pretrained(
    "/home/faith/chatglm2-6b", trust_remote_code=True)
chatglm = ChatGLM(model_new,tokenizer,max_chat_rounds=20) #支持多轮对话，可以从之前对话上下文提取知识。


# %%
text = '''已知信息：
    以下内容都是提问的设备3045的相关信息:15.照明装置设备整体结构大概可分为 部分组成整机重量 吨为方便设备运输或移动给料斗可升降尾料输送机可折叠侧面楼梯可折叠本章只记录主要零部件如想查阅更详细的零部件信息请参考KJ-3045B 机架图片的链接是: http://img.com/cf9e206b436a624a0f5f89e6f24eab16.png-3序号 10按键可以控制柴油机的速度4.03 设备形态转换当设备移动到指定地点后便可从运输状态图 4.03-1转换为工作状态图 57 -图片的链接是: http://img.com/273bdbe9b0ad0dd0fe687a4c1d0b3261.png 图 4.03-1 设备运输状态图片的链接是: http://img.com/2bf1e565bf2f16f849f65458600ee3fa.png 图 4.03-2 设备工作状态 58 - 尾料输送机工作状态/运 输状态装换一拆除运输固定装置拆除尾料输送机运输固定螺栓图 -1序号 1图片的链接是: http://img.com/baab6a53793c8430be0b03272ae3816b.png 1.尾料输送机运输固定螺栓图 -1 尾料输送机运输状态二展开尾料输送机1.展开 尾料输送机一段操作遥控器将尾料输送机一段摇杆图 -3缓缓下拨此时尾料输送机一段油7才能使柴油机停止工作手动模式下遥控器无法控制柴油机停止自动模式下只需要按一下遥控器上的柴油机停止按钮-3序号 8柴油机就会减速直至停止如需要更详细的介绍请参照柴油机控制柜使用说明书4.04 设备安装设备安装只能由我公司售后服务工程师或受过专业操作技能培训的人员获得用户授权后方可进行安装作业 

    根据上述已知信息，简洁和专业回答用户的问题，如果问题里询问图片，请返回相关的图片具体链接。如果已知信息里有多个图片，请返回最匹配的图片链接，并用[]包含链接内容而且不要有其他文字描述。
    如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：3045的设备运输状态图片'''
chatglm(text)
# chatglm('梦中情人和梦中情炉有什么区别')
# chatglm('这是个啥子意思哟:梦中情炉?')
# %%
