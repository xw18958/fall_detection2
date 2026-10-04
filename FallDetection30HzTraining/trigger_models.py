import torch
from torch import nn

class TinyTrigger(nn.Module):
    """Full 90x6 input; 3 temporal stages, no attention/BN/recurrent state."""
    def __init__(self,width=8,pooling='mean'):
        super().__init__();self.width=width;self.pooling=pooling
        if pooling not in ('mean','mean_max'):raise ValueError('Unknown temporal pooling')
        self.stem=nn.Conv1d(6,width,5,stride=2,padding=2)
        self.depthwise=nn.ModuleList([nn.Conv1d(width,width,5,stride=2,padding=2,groups=width) for _ in range(2)])
        self.pointwise=nn.ModuleList([nn.Conv1d(width,width,1) for _ in range(2)])
        self.head=nn.Linear(width*(2 if pooling=='mean_max' else 1),2)
    def forward(self,x):
        if x.shape[1:]!=(90,6):raise ValueError('Trigger requires full 90x6 input')
        h=torch.relu(self.stem(x.transpose(1,2)))
        for dw,pw in zip(self.depthwise,self.pointwise):h=torch.relu(pw(torch.relu(dw(h))))
        pooled=h.mean(-1)
        if self.pooling=='mean_max':pooled=torch.cat([pooled,h.max(-1).values],1)
        return self.head(pooled)

class TinyMLP(nn.Module):
    def __init__(self):super().__init__();self.layers=nn.Sequential(nn.Flatten(),nn.Linear(540,16),nn.ReLU(),nn.Linear(16,2))
    def forward(self,x):return self.layers(x)

def keras_equivalent(model,tf):
    """Explicit (1,time) conv geometry; same symmetric padding as Torch."""
    x=tf.keras.Input((90,6),batch_size=1);h=x
    if isinstance(model,TinyMLP):
        h=tf.keras.layers.Flatten()(h)
        for layer in (model.layers[1],model.layers[3]):
            dense=tf.keras.layers.Dense(layer.out_features,activation='relu' if layer.out_features!=2 else None)
            h=dense(h);dense.set_weights([layer.weight.detach().cpu().numpy().T,layer.bias.detach().cpu().numpy()])
    else:
        h=tf.keras.layers.Reshape((1,90,6))(h)
        def conv(h,layer,depth=False,relu=True):
            if layer.kernel_size[0]==5:h=tf.keras.layers.ZeroPadding2D(((0,0),(2,2)))(h)
            if depth:
                op=tf.keras.layers.DepthwiseConv2D((1,5),strides=(1,2),padding='valid',activation='relu')
                w=layer.weight.detach().cpu().numpy().transpose(2,0,1)[None,:,:,:]
            else:
                op=tf.keras.layers.Conv2D(layer.out_channels,(1,layer.kernel_size[0]),strides=(1,layer.stride[0]),padding='valid',activation='relu' if relu else None)
                w=layer.weight.detach().cpu().numpy().transpose(2,1,0)[None,:,:,:]
            h=op(h);op.set_weights([w,layer.bias.detach().cpu().numpy()]);return h
        h=conv(h,model.stem)
        for dw,pw in zip(model.depthwise,model.pointwise):h=conv(conv(h,dw,True),pw)
        mean=tf.keras.layers.GlobalAveragePooling2D()(h)
        if getattr(model,'pooling','mean')=='mean_max':
            maximum=tf.keras.layers.GlobalMaxPooling2D()(h)
            h=tf.keras.layers.Concatenate()([mean,maximum])
        else:h=mean
        op=tf.keras.layers.Dense(2);h=op(h);op.set_weights([model.head.weight.detach().cpu().numpy().T,model.head.bias.detach().cpu().numpy()])
    return tf.keras.Model(x,h)
