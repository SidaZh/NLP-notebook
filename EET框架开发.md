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

### 3. 代码备份

```C++
// 打印数据
std::vector<half> outputs;
outputs.clear();
outputs.resize(16);

cudaMemcpy(outputs.data(), layernormed_query.data_ptr(), sizeof(half) * 16, cudaMemcpyDeviceToHost);
printf("outputs step %d:\n",step);

for (int i = 0; i < 16; i++)
{
    std::cout << outputs[i] << " ";
}
printf("\n");
```



## 4.  EET框架

#### 2.1 基本使用

EETBertModel

​	__init__

​	__call__

​		input_ids:输入序列在词汇表中的索引

init：构造函数，使用加载预训练模型param的模型模块

call：调用前向过程

from_torch：加载torch的tensor（weight），再调用init函数



from_pretrained

layer_model_dict：	dict{分组键（网络结构名前8位）：dict{state_dict网络结构名称：对应参数tensor}}



- **tips：**
  1. GPU第一次推理速度很慢（与初始化有关），——但是EET框架算子不存在这个问题
  2. nvprof性能分析工具，
  3. NVIDIA Nsight分析工具





#### 2.2 体会

1. EET模型做微调，适配权重参数；——易用性





#### 2.3 性能优化

1. Gemm矩阵乘，cublas库，gemm择优
2. 混合精度，
3. int8量化
4. 分析瓶颈：工具nvprof、nsys