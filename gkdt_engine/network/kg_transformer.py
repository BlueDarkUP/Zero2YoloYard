import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_self_attention=True, use_cross_attention=True):
        super(TransformerBlock, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        
        # Self attention components (optional)
        if use_self_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=d_model, 
                num_heads=num_heads, 
                dropout=dropout, 
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
        
        # Cross attention components
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model, 
                num_heads=num_heads, 
                dropout=dropout, 
                batch_first=True
            )
            self.norm_ca1 = nn.LayerNorm(d_model)
            self.norm_ca2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
        
        # Feed forward components
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, cross_kv, self_attention_mask=None, cross_attention_mask=None):
        """
        Args:
            x: (B, N, C) - query序列 - batch_first
            cross_kv: (B, L, C) - cross attention的key/value (图像特征) - batch_first
            self_attention_mask: (B*num_heads, N, N) - self attention mask
            cross_attention_mask: (B*num_heads, N, L) - cross attention mask
        """
        # Self attention with residual connection (optional)
        if self.use_self_attention:
            x = self.norm1(x)
            attn_output, _ = self.self_attention(
                query=x,
                key=x,
                value=x,
                attn_mask=self_attention_mask,
                need_weights=False
            )
            x = x + self.dropout1(attn_output)
        
        # Cross attention with residual connection
        if self.use_cross_attention:
            x = self.norm_ca1(x)
            cross_kv = self.norm_ca2(cross_kv)
            cross_output, _ = self.cross_attention(
                query=x,
                key=cross_kv,
                value=cross_kv,
                attn_mask=cross_attention_mask,
                need_weights=False
            )
            x = x + self.dropout2(cross_output)
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(self.norm3(x))
        x = x + self.dropout3(ff_output)
        
        return x

class KGTransformer(nn.Module):
    def __init__(self, 
                 num_blocks=6, 
                 d_model=384, 
                 num_heads=8, 
                 d_ff=2048, 
                 output_dim=384, 
                 dropout=0.1,
                 use_self_attention=True,
                 use_cross_attention=True,
                 use_mask_token=False):
        super(KGTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.use_mask_token = use_mask_token
        
        # 如果使用mask_token，初始化mask_token参数
        if self.use_mask_token:
            self.mask_token = nn.Parameter(torch.empty(1, d_model))
            nn.init.zeros_(self.mask_token)
        
        # Create transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, use_self_attention, use_cross_attention)
            for _ in range(num_blocks)
        ])
        
        # Final layers
        self.final_norm = nn.LayerNorm(d_model)
        # self.linear_projector = nn.Linear(d_model, output_dim)
    
    def _create_self_attention_mask(self, input_mask):
        """
        将input_mask转换为self attention mask
        Args:
            input_mask: (B, N) - valid>0, invalid=0
        Returns:
            self_attention_mask: (B*num_heads, N, N) - True表示mask,False表示unmask
        """
        if input_mask is None:
            return None
            
        B, N = input_mask.shape
        
        # 创建一个head的3D mask: (B, N, N)
        mask_3d = torch.zeros(B, N, N, dtype=torch.bool, device=input_mask.device)

        invalid_positions = (input_mask == 0).nonzero(as_tuple=True)  # 获取所有invalid位置

        if len(invalid_positions[0]) > 0:
            # mask行
            mask_3d[invalid_positions[0], invalid_positions[1], :] = True
            # mask列  
            mask_3d[invalid_positions[0], :, invalid_positions[1]] = True
        
        # 创建多个head的3D mask: (B*num_heads, N, N)
        mask_3d = mask_3d.reshape(B, 1, N, N).expand(B, self.num_heads, N, N).reshape(B*self.num_heads, N, N)

        return mask_3d
    
    def _create_cross_attention_mask(self, input_mask, image_patch_length):
        """
        将input_mask转换为cross attention mask
        Args:
            input_mask: (B, N) - valid>0, invalid=0
            image_patch_length: cross attention中key/value的长度L
        Returns:
            cross_attention_mask: (B*num_heads, N, L) - True表示mask, False表示unmask
        """
        if input_mask is None:
            return None
            
        B, N = input_mask.shape
        L = image_patch_length
        
        # 创建一个head的3D mask: (B, N, L)
        mask_3d = torch.zeros(B, N, L, dtype=torch.bool, device=input_mask.device)

        # 获取所有invalid位置
        invalid_positions = (input_mask == 0).nonzero(as_tuple=True)  # 返回(batch_idx, seq_idx)

        if len(invalid_positions[0]) > 0:
            # mask行: invalid的query位置不能关注任何token
            mask_3d[invalid_positions[0], invalid_positions[1], :] = True
            
            # mask列: 无需处理，因为每个key都是valid
        
        # 创建多个head的3D mask: (B*num_heads, N, L)
        mask_3d = mask_3d.reshape(B, 1, N, L).expand(B, self.num_heads, N, L).reshape(B*self.num_heads, N, L)

        return mask_3d

    def _apply_mask_token(self, x, input_mask):
        """
        将invalid token替换为mask_token
        Args:
            x: (B, N, C) - 输入特征
            input_mask: (B, N) - valid>0, invalid=0
        Returns:
            x_masked: (B, N, C) - 替换后的特征
        """
        if input_mask is None:
            return x
            
        # 将input_mask转换为bool类型，True表示valid，False表示invalid
        masks = input_mask > 0  # (B, N)
        
        # 使用torch.where替换invalid位置的token为mask_token
        # masks.unsqueeze(-1) 扩展为 (B, N, 1) 以便广播
        x_masked = torch.where(
            masks.unsqueeze(-1), 
            x, 
            self.mask_token.to(x.dtype).unsqueeze(0)  # (1, 1, C)
        )
        
        return x_masked
        
    def forward(self, x, cross_kv, input_mask=None):
        """
        Args:
            x: Input features of shape (B, N, C)
            cross_kv: Cross attention key/value of shape (B, L, C) - 图像特征
            input_mask: Mask of shape (B, N) - valid>0, invalid=0
        
        Returns:
            Output of shape (B, N, output_dim)
        """
        B, N, C = x.shape
        B_cross, L, C_cross = cross_kv.shape
        
        assert B == B_cross, "Batch size must match between x and cross_kv"
        assert C == C_cross, "Feature dimension must match between x and cross_kv"
        
        # 根据use_mask_token选择不同的处理方式
        if self.use_mask_token:
            # 使用mask_token方式：将invalid token替换为mask_token
            if input_mask is not None:
                x = self._apply_mask_token(x, input_mask)
            
            # mask_token方式下，attention mask设为None
            self_attention_mask = None
            cross_attention_mask = None
        else:
            # 转换mask格式
            self_attention_mask = self._create_self_attention_mask(input_mask)
            cross_attention_mask = self._create_cross_attention_mask(input_mask, L)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(
                x, 
                cross_kv, 
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask
            )
        
        # Final layernorm and linear projection
        x = self.final_norm(x)
        # x = self.linear_projector(x)
        
        return x

# 测试
def test_kg_transformer():
    batch_size = 2
    seq_len = 5
    image_patch_len = 8
    d_model = 256
    
    model = KGTransformer(
        num_blocks=3,
        d_model=d_model,
        num_heads=2,
        output_dim=256
    )
    
    # 输入数据
    x = torch.randn(batch_size, seq_len, d_model)
    cross_kv = torch.randn(batch_size, image_patch_len, d_model)
    
    # input_mask: valid>0, invalid=0
    input_mask = torch.tensor([
        [1, 1, 0, 1, 1],   # 第2个位置invalid
        [0, 1, 1, 1, 0]    # 第0和第4个位置invalid
    ])
    
    output = model(x, cross_kv, input_mask)
    print("Output shape:", output.shape)  # (2, 5, 256)

if __name__ == '__main__':
    test_kg_transformer()