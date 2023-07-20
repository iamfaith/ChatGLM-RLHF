import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import inspect
import sys
print(sys.path)
import torch
from itertools import chain
import json
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from transformers import AutoTokenizer, AutoModel
# from chatglm_local.modeling_chatglm import ChatGLMForConditionalGeneration

import sys, os
# sys.path.append("/home/faith/chatglm2-6b")
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
sys.path.append(project_dir)

from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration

from models_rlhf import Critic, Reward, RewardBySimilarity
from utils_chatglm import generate_inputs
import random
from collections import defaultdict

# set device
action_device = "cuda:1"
RM_device = "cpu" #"cuda:0"
RM_device = "cuda:0"
critic_device = "cuda:0" # "cpu" 

reward_model = RewardBySimilarity(device=RM_device)

tokenizer = AutoTokenizer.from_pretrained("/home/faith/chatglm2-6b", trust_remote_code=True)
if "cuda" in action_device:
    model = ChatGLMForConditionalGeneration.from_pretrained("/home/faith/chatglm2-6b", trust_remote_code=True)
    model = model.half().cuda(action_device) # half for gpu only
elif "cpu" == action_device:
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).bfloat16()

critic = Critic(device=critic_device, m=model.transformer)

from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType, AdaLoraConfig
# Define prefix tuning config
# https://huggingface.co/docs/peft/task_guides/seq2seq-prefix-tuning
# peft_config = PrefixTuningConfig(
#     peft_type=PeftType.PREFIX_TUNING,
#     task_type=TaskType.SEQ_2_SEQ_LM,
#     inference_mode=False,
#     # prefix_length=4,
#     # prefix_dropout=0.1,
#     num_virtual_tokens=20
# )

# # Wrap model with prefix tuning
# model = get_peft_model(model, peft_config)


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

model = get_peft_model(model, peft_config)

model.is_parallelizable = True
model.model_parallel = True
model.print_trainable_parameters()
model = model.half()


#################################### old
# # 只更新embedding
# model.requires_grad_(False)
# # model.transformer.word_embeddings.requires_grad_(True)
# model.get_input_embeddings().requires_grad_(True)
#################################### old



# model.lm_head.requires_grad_(True)
# model.lm_head.weight和model.transformer.word_embeddings.weight是共享参数，两者是转置关系


decay_up_matrix_T = None
def get_decay_up_matrix_T(dtype=torch.float, device="cpu", max_length = 2048, gamma=0.99, tau=0.95):
    global decay_up_matrix_T
    if decay_up_matrix_T is None:
        # 生成衰减矩阵
        decay = gamma*tau
        decay_row = torch.ones(max_length, dtype=dtype, device=device)*decay
        decay_row[0] = 1
        decay_row_cross_time = decay_row.cumprod(dim=-1)
        assert decay_row_cross_time.sign().min() == 0
        decay_up_matrix = torch.zeros((max_length, max_length), dtype=dtype, device=device)
        for i in range(max_length):
            decay_row = decay_row_cross_time.roll(i)
            decay_row[:i] = 0 # 确保看不见前面的
            decay_up_matrix[i] = decay_row
        decay_up_matrix_T = decay_up_matrix.T# 先进行转置，因为后面需要用到矩阵乘法
    return decay_up_matrix_T

def gae_vectorize(values, rewards, masks=None):
    """
        values:表示各个时间步状态的状态值。shape:batch_size,sequence_length
        rewards:表示各个时间步做出的动作的奖励，对于gpt当前动作也是动作对应的下一状态。所以shape和values一样
                # 注意这里的rewards表示当前动作状态的reward
        masks:由于是要对生成的actions做gae，也就是泛化优势估计，
                # 所以类似以往的mask只需要对padding进行mask，
                # 因为padding的delta会被放入加权计算，而action前面的delta，
                # 由于生成的衰减矩阵就是上三角的，自然就看不到前面的。
                # 0表示mask， 1表示需要的。
    """
    action_rewards = rewards.roll(-1) # 当前状态的动作的奖励是下一个状态出现时给出的，而奖励是基于状态计算的，所以需要shift一个时间步回去
    # 为了学到最后输出的<eop>,所以给最后的状态赋予一个rewards试试
    action_rewards = (action_rewards+rewards)/2 # 将奖励分配到最后两步

    values_estimator_1_order = action_rewards + values.roll(-1) # 这里要注意roll是循环的，所以最后一位的值可能不能用
    deltas = values_estimator_1_order - values  #必须要action+下一个时刻的值函数减去当前值函数，这是表示当前action的优势
    # get weight matrix
    decay_up_matrix_T = get_decay_up_matrix_T(dtype=deltas.dtype, device= deltas.device)
    # 计算gae
    max_goal_length = deltas.shape[-1]
    sub_decay_up_matrix_T = decay_up_matrix_T[:max_goal_length, :max_goal_length]
    if masks is not None:
        deltas = deltas * masks
    gae = deltas.matmul(sub_decay_up_matrix_T.to(deltas.device))
    assert gae.shape == deltas.shape
    return gae

def get_log_prob(generated_outputs, input_ids, gen_method = "greedy_search"):
    # beam_search generate 给出来的scores就是log_prob了，所以直接gather获取即可
    gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:] 
    # let's stack the logits generated at each step to a tensor
    # 要小心greedy search 拿到的是score，需要再log_softmax
    # 而beam_search 拿到的已经是log_softmax了
    scores = torch.stack(generated_outputs.scores, dim=1)
    # if scores.max() >0 :
    #     gen_method = "greedy_search"
    if gen_method == "beam_search":
        log_prob_stacked = scores
    else:
        log_prob_stacked = torch.stack(generated_outputs.scores, dim=1).log_softmax(dim=-1)
    # now we need to collect the log_prob of the generated token # we need to add a dummy dim in the end to make gather work 
    log_prob = torch.gather(log_prob_stacked, 2, gen_sequences[:, :, None]).squeeze(-1)
    return log_prob

def get_log_probs_with_input_ids(states, gen_max_len):
    input_ids = states
    model_inputs = model.prepare_inputs_for_generation(input_ids)
    output = model(**model_inputs)  #将已经生成的序列放进去计算，再次计算得到目标action也就是后续字符的概率或者log_prob值
    ###################################old
    # logits = output.logits[:, -(gen_max_len+1):-1].log_softmax(dim=-1) # 比先softmax再log好,复杂度减小，并且解决些nan问题
    # new_log_probs = logits.gather(dim=-1, index=input_ids[:, -gen_max_len:].unsqueeze(-1)).squeeze(-1)
    ###################################old
    logits = output.logits.log_softmax(dim=-1)
    new_log_probs = logits.gather(dim=-1, index=input_ids[:, -gen_max_len:].unsqueeze(1)).squeeze(-1)

    return new_log_probs



# 这段代码是用PyTorch框架编写的，它的目的是生成一些新的文本序列，可能是用于自然语言生成或者机器翻译等任务。代码的主要步骤如下：

# - 首先，它使用model.prepare_inputs_for_generation函数，根据已经生成的部分序列（input_ids）来准备模型的输入。这个函数会根据模型的类型和参数，对输入序列进行一些必要的处理，比如添加特殊的标记符号、调整维度等。
# - 然后，它使用model函数，将模型的输入传入模型中，得到模型的输出。模型的输出是一个字典，其中包含了不同层次的信息，比如注意力权重、隐藏状态等。我们最关心的是logits，它是一个张量（tensor），表示了每个位置上每个可能的单词（或者字符）的得分（score）。得分越高，表示该单词（或者字符）出现在该位置上的可能性越大。
# - 接着，它使用log_softmax函数，对logits进行归一化处理，使得每个位置上所有单词（或者字符）的得分之和为1，并且取对数。这样做的好处是，可以避免数值溢出或者下溢的问题，并且可以方便地计算概率和对数似然（log likelihood）。
# - 最后，它使用gather函数，根据输入序列中最后gen_max_len个单词（或者字符）的索引（index），从logits中提取出相应位置上相应单词（或者字符）的对数概率（log_prob）。这样就得到了一个新的张量（new_log_probs），表示了输入序列中最后gen_max_len个单词（或者字符）出现在相应位置上的对数概率。

# 那么，gather函数具体是怎么工作的呢？gather函数是一个多索引选择方法，它可以从一个张量中按照指定的维度和索引来提取特定的元素，并组成一个新的张量。例如：

# ```python
# # # 创建一个3x3的张量
# src = torch.tensor([[1, 2, 3],
#                     [4, 5, 6],
#                     [7, 8, 9]])
# # 创建一个2x2的索引张量
# index = torch.tensor([[0, 2],
#                       [1, 0]])
# # 按照第0维（列）来提取元素
# output = torch.gather(src, 0, index)
# # output = tensor([[1, 8],
# #                  [4, 2]])
# # 按照第1维（行）来提取元素
# output = torch.gather(src, 1, index)
# # output = tensor([[1, 3],
# #                  [5, 4]])
# # ```

# 可以看到，gather函数会根据索引张量中每个位置上的值，在源张量中找到相应维度上相应位置上的元素，并将其复制到输出张量中相同位置上。输出张量和索引张量具有相同的形状。如果想要更详细地了解gather函数，请参考[PyTorch官方文档](^1^)或者[这篇博客](^3^)。

# 希望这能帮助您理解这段代码和gather函数。如果您还有其他问题，请随时提问。😊

# Source: Conversation with Bing, 7/20/2023
# (1) torch.gather — PyTorch 2.0 documentation. https://pytorch.org/docs/stable/generated/torch.gather.html.
# (2) What does the gather function do in PyTorch in layman terms. https://saturncloud.io/blog/what-does-the-gather-function-do-in-pytorch-in-layman-terms/.
# (3) What does the gather function do in pytorch in layman terms?. https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms.
# (4) PyTorch gather | What is PyTorch gather? | Examples - EDUCBA. https://www.educba.com/pytorch-gather/.
# (5) How to use PyTorch gather function for indexing? - For .... https://androidkt.com/how-to-use-pytorch-gather-function-for-indexing/.






def sample_history_from_turns(turns):
    history = [ [turn["问"], random.choice(turn["好答"])] for turn in turns ]
    return history

########################### old
# optimize_params = list(model.get_input_embeddings().parameters())+list(critic.parameters())
########################### old
optimize_params = list(model.parameters())+list(critic.parameters())

from torch.optim import Adam, AdamW
# optimizer = Adam(optimize_params, lr=1e-4, eps=1e-3)
optimizer = AdamW(optimize_params, lr=1e-2)
qa_logs = defaultdict(list)

def main(prompts_path):
    max_new_tokens = 10000
    dataset = json.loads(Path(prompts_path).read_text(encoding="utf8"))
    for epoch in range(20):
        for ix, turns in enumerate(tqdm(dataset, mininterval=1)):
            history = sample_history_from_turns(turns)
            good_answers = turns[-1]["好答"]
            bad_answers = turns[-1]["坏答"]
            history_ = history
            r = random.randint(1, 5)
            if r>3:
                query = history[-1][0]
                history_ = history[:-1]
            else:
                # 将目标句直接用RL提升或降低它的概率，得到类似finetune的效果
                query = ""
            inputs, gen_len = generate_inputs(tokenizer, query=query, history=history_)
            input_ids = inputs["input_ids"].to(action_device)
            if query != "":
                num_beams, num_return_sequences = 1, 1 # 3, 2 # set bigger if you have bigger compute memory
                num_beams, num_return_sequences = 3, 2 # set bigger if you have bigger compute memory
                assert num_beams >= num_return_sequences, "candidates num should greater than returns num"
                
                gen_method = "greedy_search" if num_beams == 1 else "beam_search" 
                generate_ = model.generate(input_ids=input_ids, do_sample=False, num_beams=num_beams, max_new_tokens=max_new_tokens,
                                    num_return_sequences=num_return_sequences, use_cache=True, num_beam_groups=1, output_scores=True,
                                    output_hidden_states=False, return_dict_in_generate=True)
                sequences = generate_.sequences
                log_probs = get_log_prob(generated_outputs=generate_, input_ids=input_ids, gen_method=gen_method)
                gen_texts = tokenizer.batch_decode(sequences[:, input_ids.shape[1]:])
                out_texts = tokenizer.batch_decode(sequences)
                qa_logs[query].extend(gen_texts)
                print("--", query, qa_logs[query], sep="\n")
            else:
                # 将目标句直接用RL提升或降低它的概率，得到类似finetune的效果
                sequences = input_ids
                with torch.no_grad():
                    log_probs = get_log_probs_with_input_ids(input_ids, gen_max_len=gen_len)
                gen_texts = [history[-1][1]]
                out_texts = tokenizer.batch_decode(sequences)
                print("目标句直接用RL提升它的概率：", out_texts)

            # compute reward for generated sequences
            reward = reward_model(gen_texts=gen_texts, good_answers=good_answers, bad_answers=bad_answers).unsqueeze(1)
            assert reward.shape == (len(gen_texts), 1), "need unsqueeze for next scatter_"
            rewards = torch.zeros_like( sequences, dtype=reward.dtype, device=reward.device)
            pad_id = tokenizer.convert_tokens_to_ids("<pad>")
            masks = ( sequences!=pad_id).long().to(RM_device)
            final_position = masks.sum(dim=-1)-1
            index=final_position.unsqueeze(-1)
            rewards.scatter_(dim=1, index=index, src=reward)
            # 确保都放到values所在的device
            rewards = torch.tensor(rewards, dtype=critic.dtype, device=critic.device)
            masks = masks.to(critic.device)
            def ppo(ppo_epochs=5, states= sequences,log_probs=log_probs, rewards=rewards, masks=masks, clip_param=0.2):
                for ppo_epoch in range(ppo_epochs):
                    # compute new log probs
                    new_log_probs = get_log_probs_with_input_ids(states, log_probs.shape[1])
                    entropy = 0 # 暂时不需要熵的约束
                    # compute value
                    # 到奖励模型和值函数模型的输入可以是一样的都是生成的序列。
                    # 生成序列同时包括state和next action
                    # prepare input for critic model
                    input_ids_critic =  states.to(critic_device)
                    values = critic(input_ids=input_ids_critic)
                    # compute gae
                    gae = gae_vectorize(values=values, rewards=rewards, masks=masks)
                    advantages = gae[:, -log_probs.shape[-1]:].to(new_log_probs.device)
                    # 计算value的估计量的偏差作为actor loss
                    # 以及ppo的actor_loss
                    value_estimator_delta = advantages
                    ratio = (new_log_probs - log_probs).exp()
                    # torch.set_printoptions(edgeitems=1)
                    # print("reward",reward, "ratio:", ratio, sep="\n")
                    if torch.isinf(ratio).any():
                        break
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
                    actor_loss  = - torch.min(surr1, surr2).mean()
                    critic_loss = value_estimator_delta.square().mean()
                    loss = 0.5 * (critic_loss + actor_loss) - 0.001 * entropy
                    # optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print("loss", loss)
            torch.cuda.empty_cache()
            ppo()

    ques = '谁是你的主人捏？'
    ques = '''已知信息：
    以下内容都是提问的设备3045的相关信息:15.照明装置设备整体结构大概可分为 部分组成整机重量 吨为方便设备运输或移动给料斗可升降尾料输送机可折叠侧面楼梯可折叠本章只记录主要零部件如想查阅更详细的零部件信息请参考KJ-3045B 机架图片的链接是: http://img.com/cf9e206b436a624a0f5f89e6f24eab16.png-3序号 10按键可以控制柴油机的速度4.03 设备形态转换当设备移动到指定地点后便可从运输状态图 4.03-1转换为工作状态图 57 -图片的链接是: http://img.com/273bdbe9b0ad0dd0fe687a4c1d0b3261.png 图 4.03-1 设备运输状态图片的链接是: http://img.com/2bf1e565bf2f16f849f65458600ee3fa.png 图 4.03-2 设备工作状态 58 - 尾料输送机工作状态/运 输状态装换一拆除运输固定装置拆除尾料输送机运输固定螺栓图 -1序号 1图片的链接是: http://img.com/baab6a53793c8430be0b03272ae3816b.png 1.尾料输送机运输固定螺栓图 -1 尾料输送机运输状态二展开尾料输送机1.展开 尾料输送机一段操作遥控器将尾料输送机一段摇杆图 -3缓缓下拨此时尾料输送机一段油7才能使柴油机停止工作手动模式下遥控器无法控制柴油机停止自动模式下只需要按一下遥控器上的柴油机停止按钮-3序号 8柴油机就会减速直至停止如需要更详细的介绍请参照柴油机控制柜使用说明书4.04 设备安装设备安装只能由我公司售后服务工程师或受过专业操作技能培训的人员获得用户授权后方可进行安装作业 

    根据上述已知信息，简洁和专业回答用户的问题，如果问题里询问图片，请返回相关的图片具体链接。如果已知信息里有多个图片，请返回最匹配的图片链接，并用[]包含链接内容而且不要有其他文字描述。
    如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：3045的设备运输状态图片'''
    input_encoded = tokenizer.batch_encode_plus([ques], return_tensors="pt", padding=True)
    # print("input_encoded = {}".format(input_encoded), flush=True)
    for t in input_encoded:
        if torch.is_tensor(input_encoded[t]):
            input_encoded[t] = input_encoded[t].to(action_device)

    outputs = model.generate(
        **input_encoded,
        num_beams=1,
        num_return_sequences=1,
        no_repeat_ngram_size=1,
        max_length = 10000,
        remove_invalid_values=True,
        max_new_tokens = max_new_tokens
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('final', response)


    response = model.chat(tokenizer, ques)


    print(response)


    # generate_ = model.generate(**input_encoded, do_sample=False, num_beams=num_beams, max_new_tokens=max_new_tokens,
    #                     num_return_sequences=num_return_sequences, use_cache=True, num_beam_groups=1, output_scores=True,
    #                     output_hidden_states=False, return_dict_in_generate=True)
    # sequences = generate_.sequences
    # # gen_texts = tokenizer.batch_decode(sequences[:, input_ids.shape[1]:])
    # out_texts = tokenizer.batch_decode(sequences)
    # print(out_texts)

def test_model(model):
    pass

if __name__ == "__main__":
    file_dir = os.path.dirname(__file__)
    dialogues_path = os.path.join(file_dir, "data", "profile_instance.json")
    dialogues_path = os.path.join(file_dir, "data", "profile_test.json")

    main(prompts_path = dialogues_path)

