## 接口

torch.flatten(t, start_dim=0, end_dim=-1) 



torch.transpose

torch.from_numpy()



**contiguous** 返回连续内存tensor深拷贝



简单的数据用`transpose()`就可以了，但是个人觉得不够直观，指向性弱了点；复杂维度的可以用`permute()`，对于维度的改变，一般更加精准；view操作的tensor要求连续排布，修改的是同一个对象。



注意映射操作时的**权重转置**关系

- Linear层的weight的shape是(out_features, in_features)，output_channel在前
- Conv1D层的weight的shape是(in_features, out_features)，权重相比较Linear层已经做了转置，乘法时不需转置







### 常用

1. tensor和numpy array相互转换

   Tensor---->Numpy  可以使用 data.numpy()，data为Tensor变量

   Numpy ----> Tensor 可以使用torch.from_numpy(data)，data为numpy变量

2. 数据类型转换

   type_as
   
   tensor.to()



- tips

torch.set_printoptions(precision=3, sci_mode=False)

input = np.load("/root/data/bert_input.npy")





## C++扩展api

torch::from_blob