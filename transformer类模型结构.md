#### ViT embedding

```python
class PatchEmbedding(nn.Layer):
    def __init__(self,
                image_size = 224,
                patch_size = 16,
                in_channels = 3,
                embed_dim = 768,
                dropout = 0.):
        super(PatchEmbedding, self).__init__()

        n_patches = (image_size // patch_size) * (image_size // patch_size) #14 * 14 = 196(个)

        self.patch_embedding = nn.Conv2D(in_channels = in_channels,
                                         out_channels = embed_dim,
                                         kernel_size = patch_size,
                                         stride = patch_size)
        
        self.dropout=nn.Dropout(dropout)

        #add class token
        self.cls_token = paddle.create_parameter(
                                        shape = [1, 1, embed_dim],
                                        dtype = 'float32',
                                        default_initializer = paddle.nn.initializer.Constant(0)
                                        #常量初始化参数，value=0， shape=[1, 1, 768]
                                        )

        #add position embedding
        self.position_embeddings = paddle.create_parameter(
                                        shape = [1, n_patches + 1, embed_dim],
                                        dtype = 'float32',
                                        default_initializer = paddle.nn.initializer.TruncatedNormal(std = 0.02)
                                        #随机截断正态（高斯）分布初始化函数
                                        )

    def forward(self, x):
        x = self.patch_embedding(x) #[N, C, H', W',]  to  [N, embed_dim, H, W]卷积层
        x = x.flatten(2)            #[N, embed_dim, H * W]
        x = x.transpose([0, 2, 1])  #[N, H * W, embed_dim]

        cls_token = self.cls_token.expand((x.shape[0], -1, -1)) #[N, 1, embed_dim]
        x = paddle.concat((cls_token, x), axis = 1)             #[N, H * W + 1, embed_dim]
        x = x + self.position_embeddings                        #[N, H * W + 1, embed_dim]
        x = self.dropout(x)

        return x
```



#### SelfAttention

```python
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)	# [N, seq_len, head, head_size] --> [N, head, seq_len, head_size]
    
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        # Q*KT / sqrt(dk)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # softmax(qk/scale)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        #
        context_layer = torch.matmul(attention_probs, value_layer)
        # [N, head, seq_len, head_size] --> [N, seq_len, head, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # [N, seq_len, head, head_size] --> [N, seq_len, embed_size] head*head_size
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) 
```



#### **CrossAttention**

```python

	query_states	# [bsz, num_heads, tgt_len, head_dim]
    key_states, value_states	# [bsz, num_heads, src_len, head_dim]
    
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))	# [bsz, num_heads, tgt_len, src_len]
	attn_probs = nn.functional.softmax(attn_weights, dim=-1)	# [bsz, num_heads, tgt_len, src_len]
    
	attn_output = torch.bmm(attn_probs, value_states)	# [bsz, num_heads, tgt_len, head_dim]
	attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)	# [bsz, tgt_len, num_heads, head_dim]
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)	# [bsz, tgt_len, embed_dim]
    
    attn_output = self.out_proj(attn_output)
    
```



T5Attention relative_attention_bias:

```python

self.relative_attention_num_buckets = config.relative_attention_num_buckets	# 32
self.relative_attention_max_distance = config.relative_attention_max_distance

if self.has_relative_attention_bias:
	self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)	

context_position = torch.arange(query_length, dtype=torch.long)[:, None]
context_position = torch.arange(key_length, dtype=torch.long)[:, None]
relative_position = memory_position - context_position	# shape (query_length, key_length)


values = self.relative_attention_bias(relative_position_bucket)	# shape (query_length, key_length, num_heads)
values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
```



### Transformer结构分类

[transformer模型分类](https://chowdera.com/2021/10/20211008192820500x.html)

1. Encoder-only 自编码 bert-like

2. Decoder-only 自回归 GPT-like

3. Encoder-Decoder seq2seq BART/T5

   input_ids->embedding->encoder->encoder_out

   decoder_ids->



参数：

**mask**

head_mask：

attention_mask：

**decoder**

past_key_value：









## Bert和ViT的Attention区别

- Bert

**SelfAttention**

attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

attention_scores = attention_scores / math.sqrt(self.attention_head_size)

attention_probs = nn.functional.softmax(attention_scores, dim=-1)

attention_probs = self.dropout(attention_probs)

context_layer = torch.matmul(attention_probs, value_layer)



**SelfOutput**

hidden_states = self.dense(hidden_states)

hidden_states = self.dropout(hidden_states)

hidden_states = self.LayerNorm(hidden_states + input_tensor)





- ViT



layernorm(hidden_states)

**SelfAttention**

attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

attention_scores = attention_scores / math.sqrt(self.attention_head_size)

attention_probs = nn.functional.softmax(attention_scores, dim=-1)

attention_probs = self.dropout(attention_probs)

context_layer = torch.matmul(attention_probs, value_layer)



**SelfOutput**

hidden_states = self.dense(hidden_states)

hidden_states = self.dropout(hidden_states)



**Attention**

hidden_states = attention_output + hidden_states

layer_output = self.layernorm_after(hidden_states)



2022年3月1日

ViT精度问题定位：

注意weight的转置关系



2022年3月16日

权重拷贝

encoder.layer.0.attention.attention.query.weight

vision_model.encoder.layers.0.self_attn.k_proj.weight

模型拼接



**CLIP模型state_dict**

**text_model**

logit_scale
text_model.embeddings.position_ids
text_model.embeddings.token_embedding.weight
text_model.embeddings.position_embedding.weight

text_model.encoder.layers.0.self_attn.k_proj.weight
text_model.encoder.layers.0.self_attn.k_proj.bias
text_model.encoder.layers.0.self_attn.v_proj.weight
text_model.encoder.layers.0.self_attn.v_proj.bias
text_model.encoder.layers.0.self_attn.q_proj.weight
text_model.encoder.layers.0.self_attn.q_proj.bias
text_model.encoder.layers.0.self_attn.out_proj.weight
text_model.encoder.layers.0.self_attn.out_proj.bias
text_model.encoder.layers.0.layer_norm1.weight
text_model.encoder.layers.0.layer_norm1.bias
text_model.encoder.layers.0.mlp.fc1.weight
text_model.encoder.layers.0.mlp.fc1.bias
text_model.encoder.layers.0.mlp.fc2.weight
text_model.encoder.layers.0.mlp.fc2.bias
text_model.encoder.layers.0.layer_norm2.weight
text_model.encoder.layers.0.layer_norm2.bias

text_model.final_layer_norm.weight
text_model.final_layer_norm.bias

text_projection.weight

**vision_model**

vision_model.embeddings.class_embedding
vision_model.embeddings.position_ids
vision_model.embeddings.patch_embedding.weight
vision_model.embeddings.position_embedding.weight

vision_model.pre_layrnorm.weight
vision_model.pre_layrnorm.bias

vision_model.encoder.layers.0.self_attn.k_proj.weight
vision_model.encoder.layers.0.self_attn.k_proj.bias
vision_model.encoder.layers.0.self_attn.v_proj.weight
vision_model.encoder.layers.0.self_attn.v_proj.bias
vision_model.encoder.layers.0.self_attn.q_proj.weight
vision_model.encoder.layers.0.self_attn.q_proj.bias
vision_model.encoder.layers.0.self_attn.out_proj.weight
vision_model.encoder.layers.0.self_attn.out_proj.bias
vision_model.encoder.layers.0.layer_norm1.weight
vision_model.encoder.layers.0.layer_norm1.bias
vision_model.encoder.layers.0.mlp.fc1.weight
vision_model.encoder.layers.0.mlp.fc1.bias
vision_model.encoder.layers.0.mlp.fc2.weight
vision_model.encoder.layers.0.mlp.fc2.bias
vision_model.encoder.layers.0.layer_norm2.weight
vision_model.encoder.layers.0.layer_norm2.bias

vision_model.post_layernorm.weight
vision_model.post_layernorm.bias
visual_projection.weight



**ViT模型state_dict**

embeddings.cls_token
embeddings.position_embeddings
embeddings.patch_embeddings.projection.weight
embeddings.patch_embeddings.projection.bias
encoder.layer.0.attention.attention.query.weight
encoder.layer.0.attention.attention.query.bias
encoder.layer.0.attention.attention.key.weight
encoder.layer.0.attention.attention.key.bias
encoder.layer.0.attention.attention.value.weight
encoder.layer.0.attention.attention.value.bias
encoder.layer.0.attention.output.dense.weight
encoder.layer.0.attention.output.dense.bias
encoder.layer.0.intermediate.dense.weight
encoder.layer.0.intermediate.dense.bias
encoder.layer.0.output.dense.weight
encoder.layer.0.output.dense.bias
encoder.layer.0.layernorm_before.weight
encoder.layer.0.layernorm_before.bias
encoder.layer.0.layernorm_after.weight
encoder.layer.0.layernorm_after.bias

layernorm.weight
layernorm.bias
pooler.dense.weight
pooler.dense.bias





**Bert模型 state_dict**

embeddings.position_ids
embeddings.word_embeddings.weight
embeddings.position_embeddings.weight
embeddings.token_type_embeddings.weight
embeddings.LayerNorm.weight
embeddings.LayerNorm.bias

layer.0.attention.self.query.weight
layer.0.attention.self.query.bias
layer.0.attention.self.key.weight
layer.0.attention.self.key.bias
layer.0.attention.self.value.weight
layer.0.attention.self.value.bias
layer.0.attention.output.dense.weight
layer.0.attention.output.dense.bias
layer.0.attention.output.LayerNorm.weight
layer.0.attention.output.LayerNorm.bias
layer.0.intermediate.dense.weight
layer.0.intermediate.dense.bias
layer.0.output.dense.weight
layer.0.output.dense.bias
layer.0.output.LayerNorm.weight
layer.0.output.LayerNorm.bias



**T5模型 state_dict**

shared.weight
encoder.embed_tokens.weight
encoder.block.0.layer.0.SelfAttention.q.weight
encoder.block.0.layer.0.SelfAttention.k.weight
encoder.block.0.layer.0.SelfAttention.v.weight
encoder.block.0.layer.0.SelfAttention.o.weight
encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight
encoder.block.0.layer.0.layer_norm.weight
encoder.block.0.layer.1.DenseReluDense.wi.weight
encoder.block.0.layer.1.DenseReluDense.wo.weight
encoder.block.0.layer.1.layer_norm.weight

encoder.final_layer_norm.weight
decoder.embed_tokens.weight
decoder.block.0.layer.0.SelfAttention.q.weight
decoder.block.0.layer.0.SelfAttention.k.weight
decoder.block.0.layer.0.SelfAttention.v.weight
decoder.block.0.layer.0.SelfAttention.o.weight
decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight
decoder.block.0.layer.0.layer_norm.weight
decoder.block.0.layer.1.EncDecAttention.q.weight
decoder.block.0.layer.1.EncDecAttention.k.weight
decoder.block.0.layer.1.EncDecAttention.v.weight
decoder.block.0.layer.1.EncDecAttention.o.weight
decoder.block.0.layer.1.layer_norm.weight
decoder.block.0.layer.2.DenseReluDense.wi.weight
decoder.block.0.layer.2.DenseReluDense.wo.weight
decoder.block.0.layer.2.layer_norm.weight

decoder.final_layer_norm.weight

**注意：**Google的T5没有除以根号dk的，但它依然能够正常收敛，那是因为它在初始化策略上做了些调整，所以这个事情还跟初始化有关



**Albert模型state_dict**

embeddings.position_ids
embeddings.word_embeddings.weight
embeddings.position_embeddings.weight
embeddings.token_type_embeddings.weight
embeddings.LayerNorm.weight
embeddings.LayerNorm.bias
encoder.embedding_hidden_mapping_in.weight
encoder.embedding_hidden_mapping_in.bias

encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight
encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias
encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight
encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias
encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight
encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias
encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight
encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias
encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight
encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias
encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight
encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias
encoder.albert_layer_groups.0.albert_layers.0.ffn.weight
encoder.albert_layer_groups.0.albert_layers.0.ffn.bias
encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight
encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias
pooler.weight
pooler.bias



**GPT模型 state_dict**

wte.weight
wpe.weight

h.0.ln_1.weight
h.0.ln_1.bias
h.0.attn.bias
h.0.attn.masked_bias
h.0.attn.c_attn.weight
h.0.attn.c_attn.bias
h.0.attn.c_proj.weight
h.0.attn.c_proj.bias
h.0.ln_2.weight
h.0.ln_2.bias
h.0.mlp.c_fc.weight
h.0.mlp.c_fc.bias
h.0.mlp.c_proj.weight
h.0.mlp.c_proj.bias

ln_f.weight
ln_f.bias



**Bart模型state dict**

shared.weight
encoder.embed_tokens.weight
encoder.embed_positions.weight
encoder.layers.0.self_attn.k_proj.weight
encoder.layers.0.self_attn.k_proj.bias
encoder.layers.0.self_attn.v_proj.weight
encoder.layers.0.self_attn.v_proj.bias
encoder.layers.0.self_attn.q_proj.weight
encoder.layers.0.self_attn.q_proj.bias
encoder.layers.0.self_attn.out_proj.weight
encoder.layers.0.self_attn.out_proj.bias
encoder.layers.0.self_attn_layer_norm.weight
encoder.layers.0.self_attn_layer_norm.bias
encoder.layers.0.fc1.weight
encoder.layers.0.fc1.bias
encoder.layers.0.fc2.weight
encoder.layers.0.fc2.bias
encoder.layers.0.final_layer_norm.weight
encoder.layers.0.final_layer_norm.bias

encoder.layernorm_embedding.weight
encoder.layernorm_embedding.bias

decoder.embed_tokens.weight
decoder.embed_positions.weight
decoder.layers.0.self_attn.k_proj.weight
decoder.layers.0.self_attn.k_proj.bias
decoder.layers.0.self_attn.v_proj.weight
decoder.layers.0.self_attn.v_proj.bias
decoder.layers.0.self_attn.q_proj.weight
decoder.layers.0.self_attn.q_proj.bias
decoder.layers.0.self_attn.out_proj.weight
decoder.layers.0.self_attn.out_proj.bias
decoder.layers.0.self_attn_layer_norm.weight
decoder.layers.0.self_attn_layer_norm.bias
decoder.layers.0.encoder_attn.k_proj.weight
decoder.layers.0.encoder_attn.k_proj.bias
decoder.layers.0.encoder_attn.v_proj.weight
decoder.layers.0.encoder_attn.v_proj.bias
decoder.layers.0.encoder_attn.q_proj.weight
decoder.layers.0.encoder_attn.q_proj.bias
decoder.layers.0.encoder_attn.out_proj.weight
decoder.layers.0.encoder_attn.out_proj.bias
decoder.layers.0.encoder_attn_layer_norm.weight
decoder.layers.0.encoder_attn_layer_norm.bias
decoder.layers.0.fc1.weight
decoder.layers.0.fc1.bias
decoder.layers.0.fc2.weight
decoder.layers.0.fc2.bias
decoder.layers.0.final_layer_norm.weight
decoder.layers.0.final_layer_norm.bias

decoder.layernorm_embedding.weight
decoder.layernorm_embedding.bias

