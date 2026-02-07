# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
import torch.nn.functional as F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)  #  use torch compile by default
            else verl_F.entropy_from_logits)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # 确保处于训练模式
        self.actor_module.train()

        # 通过检查数据中是否包含 DPO 特有的 'rejected_input_ids' 字段来判断
        available_keys = data.batch.keys()
        is_dpo = 'rejected_input_ids' in available_keys

        temperature = data.meta_info['temperature']

        # --- [2. 动态构建 select_keys] ---
        # 公共基础字段
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        
        if is_dpo:
            # DPO 模式：需要成对数据 (Rejected) 和 Reference LogProb
            select_keys.extend([
                'rejected_input_ids', 'rejected_attention_mask', 'rejected_position_ids',
                'rejected_responses', 'ref_log_prob', 'rejected_ref_log_prob'
            ])
        else:
            # PPO 模式：需要旧策略的 LogProb 和优势函数
            select_keys.extend(['old_log_probs', 'advantages'])
            if self.config.use_kl_loss:
                select_keys.append('ref_log_prob')

        # 根据 keys 筛选数据
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # --- [3. 切分 Mini-Batch] ---
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        
        # --- [4. 训练循环] ---
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                
                # --- [5. Micro-Batch 切分逻辑] ---
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    # 注意：rearrange_micro_batches 需要根据你的代码库具体实现导入
                    # 这里假设上下文已存在该函数
                    from verl.utils.seqlen_balancing import rearrange_micro_batches
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # 移动数据到 GPU
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())

                    if is_dpo:
                        # 1. 提取数据
                        chosen_input_ids = data['input_ids']
                        chosen_attention_mask = data['attention_mask']
                        chosen_position_ids = data['position_ids']
                        chosen_responses = data['responses']
                        
                        rejected_input_ids = data['rejected_input_ids']
                        rejected_attention_mask = data['rejected_attention_mask']
                        rejected_position_ids = data['rejected_position_ids']
                        
                        # 2. 拼接 Chosen 和 Rejected (Batch Size * 2)
                        # 这样做是为了利用 GPU 并行，一次 Forward 计算所有 LogProb
                        concat_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
                        concat_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
                        concat_position_ids = torch.cat([chosen_position_ids, rejected_position_ids], dim=0)
                        
                        # 3. 前向传播 (Forward)
                        # 注意：这里我们手动实现了 _forward_micro_batch 的部分逻辑，以便处理拼接数据
                        output = self.actor_module(input_ids=concat_input_ids, 
                                                   attention_mask=concat_attention_mask,
                                                   position_ids=concat_position_ids)
                        
                        # 4. 手动计算 LogProbs (Shift & Gather)
                        # logits 预测下一个 token，所以需要错一位
                        shift_logits = output.logits[..., :-1, :].contiguous()
                        shift_labels = concat_input_ids[..., 1:].contiguous()
                        
                        log_probs = F.log_softmax(shift_logits, dim=-1)
                        # [2*Micro_BS, Seq_Len-1]
                        token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                        
                        # 5. Masking (复用 PPO 逻辑: 切片 Attention Mask)
                        # responses 包含了生成的 token，位于 input_ids 的末尾
                        # 我们使用 Attention Mask 的最后 len(response) 部分作为 Mask
                        response_len = chosen_responses.shape[1]
                        
                        # [2*Micro_BS, Resp_Len]
                        # 对应 shift 后的 log_probs，取最后 response_len 长度
                        response_mask = concat_attention_mask[:, -response_len:]
                        token_log_probs_resp = token_log_probs[:, -response_len:]

                        # 求和：只计算 Response 部分的 LogProb
                        # Pad 部分会被 response_mask (为0) 过滤掉
                        sum_log_probs = (token_log_probs_resp * response_mask).sum(dim=-1)

                        # 6. 拆分回 Chosen / Rejected
                        micro_bs = chosen_input_ids.shape[0]
                        policy_chosen_logps = sum_log_probs[:micro_bs]
                        policy_rejected_logps = sum_log_probs[micro_bs:]

                        # 7. 处理 Reference LogProb (从 Trainer 传入)
                        # 假设 Trainer 传过来的是 [B, Seq_Len] 的 token-level logprob
                        ref_chosen_token_logps = data['ref_log_prob']
                        ref_rejected_token_logps = data['rejected_ref_log_prob']
                        
                        # 同样的切片、Mask、求和逻辑
                        # 需要将拼接的 response_mask 拆回两部分
                        chosen_resp_mask = response_mask[:micro_bs]
                        rejected_resp_mask = response_mask[micro_bs:]
                        
                        # 注意 ref_log_prob 也是全长度的，取最后一段
                        ref_chosen_logps = (ref_chosen_token_logps[:, -response_len:] * chosen_resp_mask).sum(dim=-1)
                        ref_rejected_logps = (ref_rejected_token_logps[:, -response_len:] * rejected_resp_mask).sum(dim=-1)

                        # 8. 计算 DPO Loss
                        # 尝试从 config 获取 beta，默认 0.1
                        beta = getattr(self.config, 'beta', 0.1)
                        
                        pi_logratios = policy_chosen_logps - policy_rejected_logps
                        ref_logratios = ref_chosen_logps - ref_rejected_logps
                        logits = pi_logratios - ref_logratios
                        
                        # Loss = -log(sigmoid(beta * logits))
                        losses = -F.logsigmoid(beta * logits)
                        policy_loss = losses.mean()

                        # 9. 记录 DPO 指标
                        with torch.no_grad():
                            chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
                            rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
                            reward_acc = (chosen_rewards > rejected_rewards).float().mean()
                            reward_margin = (chosen_rewards - rejected_rewards).mean()

                        # 处理梯度累积
                        if self.config.use_dynamic_bsz:
                            loss = policy_loss * (len(data['input_ids']) / self.config.ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation
                        
                        loss.backward()

                        data_metrics = {
                            'actor/dpo_loss': policy_loss.detach().item(),
                            'actor/chosen_reward': chosen_rewards.mean().item(),
                            'actor/rejected_reward': rejected_rewards.mean().item(),
                            'actor/reward_acc': reward_acc.item(),
                            'actor/reward_margin': reward_margin.item(),
                        }
                        append_to_dict(metrics, data_metrics)

                    else:
                        responses = data['responses']
                        response_length = responses.size(1)
                        attention_mask = data['attention_mask']
                        response_mask = attention_mask[:, -response_length:]
                        old_log_prob = data['old_log_probs']
                        advantages = data['advantages']

                        clip_ratio = self.config.clip_ratio
                        entropy_coeff = self.config.entropy_coeff

                        # PPO 复用了封装好的 _forward_micro_batch
                        entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                      log_prob=log_prob,
                                                                                      advantages=advantages,
                                                                                      eos_mask=response_mask,
                                                                                      cliprange=clip_ratio)
                        
                        entropy_loss = verl_F.masked_mean(entropy, response_mask)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_prob = data['ref_log_prob']
                            kld = core_algos.kl_penalty(logprob=log_prob,
                                                        ref_logprob=ref_log_prob,
                                                        kl_penalty=self.config.kl_loss_type)
                            kl_loss = masked_mean(kld, response_mask)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics['actor/kl_loss'] = kl_loss.detach().item()
                            metrics['actor/kl_coef'] = self.config.kl_loss_coef

                        if self.config.use_dynamic_bsz:
                            loss = policy_loss * (len(data['input_ids']) / self.config.ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation
                        loss.backward()

                        data_metrics = {
                            'actor/entropy_loss': entropy_loss.detach().item(),
                            'actor/pg_loss': pg_loss.detach().item(),
                            'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                            'actor/ppo_kl': ppo_kl.detach().item(),
                        }
                        append_to_dict(metrics, data_metrics)

                grad_norm = self._optimizer_step()
                data_metrics = {'actor/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, data_metrics)
                
        self.actor_optimizer.zero_grad()
        return metrics