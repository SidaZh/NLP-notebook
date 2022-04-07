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







**2022年3月21日**

1. transformers Hugging face transformers框架CLIP模型原生不支持fp16（因为text model的causal_attention_mask为float32）
2. 