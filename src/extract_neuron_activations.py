"""Extract neuron activations from GPT2, GPTNeoX, or LLaMA models."""

import torch
try:
  # For transformers library backward compatibility.
  from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
except:
  pass


def get_attention_mask(attention_mask):
  batch_size = attention_mask.shape[0]
  attention_mask = attention_mask.view(batch_size, -1)
  attention_mask = attention_mask[:, None, None, :]
  attention_mask = attention_mask.to(dtype=torch.float32)  # fp16 compatibility
  attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
  return attention_mask


def get_position_ids(input_shape):
  position_ids = torch.arange(0, input_shape[-1] + 0, dtype=torch.long)
  position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
  return position_ids


def prepare_inputs_for_generation(input_ids,
                                  past_key_values=None,
                                  inputs_embeds=None,
                                  **kwargs):
  """Implements the same preprocessing as GPT2LMHeadModel.prepare_inputs_for_generation"""
  token_type_ids = kwargs.get("token_type_ids", None)
  # only last token for inputs_ids if past is defined in kwargs
  if past_key_values:
    input_ids = input_ids[:, -1].unsqueeze(-1)
    if token_type_ids is not None:
      token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
  attention_mask = kwargs.get("attention_mask", None)
  position_ids = kwargs.get("position_ids", None)
  if attention_mask is not None and position_ids is None:
    # create position_ids on the fly for batch generation
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if past_key_values:
      position_ids = position_ids[:, -1].unsqueeze(-1)
  else:
    position_ids = None
  # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
  if inputs_embeds is not None and past_key_values is None:
    model_inputs = {"inputs_embeds": inputs_embeds}
  else:
    model_inputs = {"input_ids": input_ids}
  model_inputs.update({
      "past_key_values": past_key_values,
      "use_cache": kwargs.get("use_cache"),
      "position_ids": position_ids,
      "attention_mask": attention_mask,
      "token_type_ids": token_type_ids,
  })
  return model_inputs


def run_block_gpt2(block, hidden_states, attention_mask, head_mask, use_cache):
  outputs = {}
  residual = hidden_states
  outputs['block_input'] = residual.detach()
  hidden_states = block.ln_1(hidden_states)
  attn_outputs = block.attn(
      hidden_states,
      layer_past=None,
      attention_mask=attention_mask,
      head_mask=head_mask,
      use_cache=use_cache,
      output_attentions=False,
  )
  attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
  outputs['attention_output'] = attn_output.detach()
  # residual connection
  hidden_states = attn_output + residual
  residual = hidden_states
  outputs['mlp_input'] = residual.detach()
  hidden_states = block.ln_2(hidden_states)
  hidden_states = block.mlp.c_fc(hidden_states)
  activations = block.mlp.act(hidden_states)
  outputs['mlp_activation'] = activations.detach()
  mlp_hidden_states = block.mlp.c_proj(activations)
  mlp_hidden_states = block.mlp.dropout(mlp_hidden_states)
  outputs['mlp_output'] = mlp_hidden_states.detach()
  mlp_hidden_states = residual + mlp_hidden_states
  outputs['block_output'] = mlp_hidden_states.detach()
  return outputs


def get_neuron_activations(model, encoded_inputs, layer_id):
  for k in encoded_inputs:
    encoded_inputs[k] = encoded_inputs[k].to(model.device)
  attention_mask = get_attention_mask(encoded_inputs['attention_mask'])
  inputs_embeds = model.wte(encoded_inputs['input_ids'])
  position_ids = prepare_inputs_for_generation(
      encoded_inputs['input_ids'],
      attention_mask=encoded_inputs['attention_mask'])['position_ids']
  position_embeds = model.wpe(position_ids)
  hidden_states = inputs_embeds + position_embeds
  hidden_states = model.drop(hidden_states)

  past_key_values = tuple([None] * len(model.h))
  head_mask = model.get_head_mask(None, model.config.n_layer)
  for i, (block, layer_past) in enumerate(zip(model.h, past_key_values)):
    if i == layer_id:
      break
    outputs = block(
        hidden_states,
        layer_past=layer_past,  # None
        attention_mask=attention_mask,
        head_mask=head_mask[i],
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=model.config.use_cache,
        output_attentions=False,
    )
    hidden_states = outputs[0]
  # ith layer
  outputs = run_block_gpt2(model.h[layer_id], hidden_states, attention_mask,
                           head_mask[layer_id], model.config.use_cache)
  if layer_id == len(model.h) - 1:
    outputs['last_layer_output'] = model.ln_f(outputs['block_output'])
  return outputs


def get_neuron_activations_batched(model,
                                   encoded_inputs,
                                   layer_id,
                                   batch_size=8):
  outputs = collections.defaultdict(list)
  for b_i in range(0, encoded_inputs['input_ids'].shape[0], batch_size):
    encoded_inputs_batch = {}
    for k in encoded_inputs:
      encoded_inputs_batch[k] = encoded_inputs[k][b_i:b_i + batch_size]
    intermediate_output_batch = get_neuron_activations(model,
                                                       encoded_inputs_batch,
                                                       layer_id)
    for k in intermediate_output_batch:
      outputs[k].append(intermediate_output_batch[k])
  return {k: torch.cat(outputs[k]) for k in outputs}


def get_representations_across_layers_gpt2(model, encoded_inputs, layer_index):
  for k in encoded_inputs:
    encoded_inputs[k] = encoded_inputs[k].to(model.device)
  attention_mask = get_attention_mask(encoded_inputs['attention_mask'])
  inputs_embeds = model.wte(encoded_inputs['input_ids'])
  position_ids = prepare_inputs_for_generation(
      encoded_inputs['input_ids'],
      attention_mask=encoded_inputs['attention_mask'])['position_ids']
  position_embeds = model.wpe(position_ids)
  hidden_states = inputs_embeds + position_embeds
  hidden_states = model.drop(hidden_states)

  past_key_values = tuple([None] * len(model.h))
  head_mask = model.get_head_mask(None, model.config.n_layer)
  all_layer_outputs = {}
  for i, (block, layer_past) in enumerate(zip(model.h, past_key_values)):
    outputs = run_block_gpt2(block, hidden_states, attention_mask, head_mask[i],
                             model.config.use_cache)
    hidden_states = outputs['block_output']
    for k in outputs:
      all_layer_outputs[f'layer_{i}-{k}'] = outputs[k]
    if i == layer_index:
      break
  if layer_index == len(model.h) - 1:
    all_layer_outputs['last_layer_output'] = model.ln_f(outputs['block_output'])
  return all_layer_outputs


def run_block_llama(block, hidden_states, attention_mask, head_mask, use_cache,
                    position_ids):
  outputs = {}
  residual = hidden_states
  outputs['block_input'] = residual.detach()
  hidden_states = block.input_layernorm(hidden_states)
  hidden_states, self_attn_weights, present_key_value = block.self_attn(
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_value=None,
      output_attentions=False,
      use_cache=use_cache,
  )
  outputs['attention_output'] = hidden_states.detach()
  hidden_states = residual + hidden_states
  residual = hidden_states
  outputs['mlp_input'] = residual.detach()
  hidden_states = block.post_attention_layernorm(hidden_states)
  activations = block.mlp.act_fn(block.mlp.gate_proj(hidden_states))
  outputs['mlp_activation'] = activations.detach()
  hidden_states = activations * block.mlp.up_proj(hidden_states)
  hidden_states = block.mlp.down_proj(hidden_states)
  outputs['mlp_output'] = hidden_states.detach()
  hidden_states = residual + hidden_states
  outputs['block_output'] = hidden_states.detach()
  return outputs


def get_representations_across_layers_llama(model, encoded_inputs, layer_index):
  for k in encoded_inputs:
    encoded_inputs[k] = encoded_inputs[k].to(model.device)
  # Prepare input arguments
  position_ids = prepare_inputs_for_generation(
      encoded_inputs['input_ids'],
      attention_mask=encoded_inputs['attention_mask'])['position_ids']
  inputs_embeds = model.embed_tokens(encoded_inputs['input_ids'])
  attention_mask_fn = model._prepare_decoder_attention_mask if hasattr(
      model,
      '_prepare_decoder_attention_mask') else _prepare_4d_causal_attention_mask
  attention_mask = attention_mask_fn(encoded_inputs['attention_mask'],
                                     encoded_inputs['input_ids'].shape,
                                     inputs_embeds, 0)
  attention_mask[attention_mask < 0] = -1000
  # Decoder layers
  hidden_states = inputs_embeds
  all_layer_outputs = {}
  for i, decoder_layer in enumerate(model.layers):
    layer_outputs = run_block_llama(decoder_layer, hidden_states,
                                    attention_mask, None,
                                    model.config.use_cache, position_ids)
    hidden_states = layer_outputs['block_output']
    for k in layer_outputs:
      all_layer_outputs[f'layer_{i}-{k}'] = layer_outputs[k]
    if i == layer_index:
      break
  if layer_index >= len(model.layers) - 1:
    all_layer_outputs['last_layer_output'] = model.norm(hidden_states)
  return all_layer_outputs


def run_block_gptneox(block,
                      hidden_states,
                      attention_mask,
                      position_ids,
                      use_cache,
                      output_attentions=False):
  outputs = {}
  outputs['block_input'] = hidden_states.detach()
  attention_input = block.input_layernorm(hidden_states)
  outputs['attention_input'] = attention_input.detach()
  attn_outputs = block.attention(
      hidden_states=attention_input,
      attention_mask=attention_mask,
      position_ids=position_ids,
      layer_past=None,
      use_cache=use_cache,
      output_attentions=output_attentions,
  )
  if output_attentions:
    outputs['attention_weights'] = attn_outputs[2].detach()
  outputs['attention_output'] = attn_outputs[0].detach()
  attn_output = block.post_attention_dropout(attn_outputs[0])

  mlp_input = block.post_attention_layernorm(hidden_states)
  outputs['mlp_input'] = mlp_input.detach()
  mlp_activation = block.mlp.dense_h_to_4h(mlp_input)
  mlp_activation = block.mlp.act(mlp_activation)
  outputs['mlp_activation'] = mlp_activation.detach()
  mlp_output = block.mlp.dense_4h_to_h(mlp_activation)
  outputs['mlp_output'] = mlp_output.detach()

  mlp_output = block.post_mlp_dropout(mlp_output)
  hidden_states = mlp_output + attn_output + hidden_states
  outputs['block_output'] = hidden_states.detach()
  return outputs


def get_representations_across_layers_gptneox(model,
                                              encoded_inputs,
                                              layer_index,
                                              feature_types=None):
  for k in encoded_inputs:
    encoded_inputs[k] = encoded_inputs[k].to(model.device)
  # Prepare input arguments
  position_ids = prepare_inputs_for_generation(
      encoded_inputs['input_ids'],
      attention_mask=encoded_inputs['attention_mask'])['position_ids']
  attention_mask = encoded_inputs['attention_mask'][:, None, None, :]
  attention_mask = attention_mask.to(dtype=model.dtype)  # fp16 compatibility
  attention_mask = (1.0 - attention_mask) * torch.finfo(model.dtype).min
  # For computation stability, i.e., overflow due to float16.
  # attention_mask[attention_mask < 0] = -1000

  inputs_embeds = model.embed_in(encoded_inputs['input_ids'])
  # decoder layers
  hidden_states = model.emb_dropout(inputs_embeds)
  all_layer_outputs = {}
  for i, decoder_layer in enumerate(model.layers):
    layer_outputs = run_block_gptneox(
        decoder_layer,
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=model.config.use_cache,
        output_attentions=model.config.output_attentions)
    hidden_states = layer_outputs['block_output']
    for k in layer_outputs:
      if feature_types and k not in feature_types:
        continue
      all_layer_outputs[f'layer_{i}-{k}'] = layer_outputs[k]
    if i == layer_index:
      break
  if layer_index >= len(model.layers) - 1:
    all_layer_outputs['last_layer_output'] = model.final_layer_norm(
        hidden_states)
  return all_layer_outputs


# Test
def test_gptnoex_features(model, encoded_inputs):
  with torch.no_grad():
    last_hidden_state = model.gpt_neox(**encoded_inputs).last_hidden_state
    all_layer_outputs = get_representations_across_layers_gptneox(
        model.gpt_neox,
        encoded_inputs,
        model.config.num_hidden_layers,
        feature_types='block_output')
    last_hidden_state_extracted = all_layer_outputs['last_layer_output']
  return torch.all(last_hidden_state == last_hidden_state_extracted)
