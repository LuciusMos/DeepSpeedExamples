# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn


# Note that the following code is modified from
# https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False,
                print_msg=None):
        loss = None

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)
        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        # rewards /= 10.  # Bloomz model spcified, because rewards output is like 50+
        # if print_msg:
        #     print(print_msg, 'reward_model inner output')
        #     print('(last_)hidden_states', hidden_states.shape, hidden_states)
        #     print('rewards', rewards.shape, rewards)

        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens(chosen_id != rejected_id) before padding
        loss = 0
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            # get padding token indices
            c_inds = (chosen_id == self.PAD_ID).nonzero()
            # OPT model pads the first token, so we need to use the seoncd padding token as the end of the sequence
            c_ind = c_inds[self.num_padding_at_beginning].item() if \
                len(c_inds) > self.num_padding_at_beginning else seq_len
            # get different tokens' indices between chosen_id and reject_id
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                # chosen_id and reject_id are exactly the same
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_inds = c_inds
                r_ind = c_ind
            else:
                # chosen_id and reject_id have differences
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item() if \
                    len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0].item()
            assert divergence_ind >= self.num_padding_at_beginning   # issue#338 reports bloomz should be >= 0

            # if print_msg:
            #     print('chosen c_ind:{}'.format(c_ind))
            #     print('reject r_ind:{}'.format(r_ind))

            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(chosen_reward[c_ind - 1])  # use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            # loss_type = "log-sig"  # or "log-sig"
            # loss_minus = c_truncated_reward - r_truncated_reward
            # if loss_type == "log-sig":
            #     loss_sig = torch.sigmoid(loss_minus)
            #     loss_log = -torch.log(loss_sig)
            # elif loss_type == "log-exp":
            #     loss_exp = torch.exp(loss_minus) + 1
            #     loss_log = torch.log(loss_exp)
            # nan_inf_mask = loss_log.isinf() | loss_log.isnan()
            # if not nan_inf_mask.any():
            #     loss += loss_log.mean()

            # # print('c/r_reward[{}:{}], c_ind:{}, r_ind:{}'.format(divergence_ind, end_ind, c_ind, r_ind))

            # # if print_msg:
            # if nan_inf_mask.any():
            #     # print('check_divergence:{}, end_ind:{}'.format(check_divergence.squeeze(), end_ind))  # noqa
            #     # print('chosen c_ind:{}, c_truncated_reward:{}, r_end_score:{}'.format(c_ind, c_truncated_reward, chosen_reward[c_ind - 1]))  # noqa
            #     # print('reject r_ind:{}, r_truncated_reward:{}, r_end_score:{}'.format(r_ind, r_truncated_reward, rejected_reward[r_ind - 1]))  # noqa
            #     print('loss mask:{}'.format(nan_inf_mask))
            #     print('chosen:{}'.format(c_truncated_reward[nan_inf_mask]))
            #     print('reject:{}', format(r_truncated_reward[nan_inf_mask]))
            #     print('minus: {}'.format(loss_minus[nan_inf_mask]))
            #     if loss_type == 'log-sig':
            #         print('[[sig: {}'.format(loss_sig[nan_inf_mask]))
            #     elif loss_type == "log-exp":
            #         print('[[exp: {}'.format(loss_exp[nan_inf_mask]))
            #     print('log:   {}'.format(loss_log[nan_inf_mask]))

            # use `softplus`` fn for numerical stability
            loss += nn.functional.softplus(r_truncated_reward - c_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        rm_ret = {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }
        # if print_msg:
        #     print('reward_model output', rm_ret)
        print('reward_model output', {k: rm_ret[k].detach().cpu().numpy() for k in rm_ret})
        return rm_ret

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence
                # so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
