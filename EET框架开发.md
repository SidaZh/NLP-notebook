## 1. 特性背景

EET适配模型时发现的一些问题：

EET接入一个新模型，需要根据预训练模型的参数名称重新定义python op封装，但是encoder decoder；也不便于用户自定义EET模型；

（1）EET python层封装的模块无法复用：比如有两个模型的结构是类似的，，通过外部参数控制，而不是侵入式修改；此外，模型参数导入的差异应该和模型结构的定义解耦开。

（2）统一基本模块命名规范的另一个好处：便于寻址；下一步开发精度比对工具时可以利用mapping比对子模型输入输出

（3）降低接入新模型的难度，提高复用性，提高模块拼装的易用性（encoder，decoder封装一个）



解决办法：

（1）在模型外部处理参数

模型加载和结构定义解耦：

EET结构命名标准化







### 2. 算子

MultiHeadAttention

```C++
if (pre_layernorm) 
{
    layer_norm
	qkv_weights_mul
} 
else 
{
    qkv_weights_mul
}

qkv_add_bias
q_k_mul
qk_softmax
attn_v_mul
transpose
project

    


```

​	

