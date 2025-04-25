import cv2
import numpy as np
import range
import torch
from PIL import Image
from builtins import range


from nets.cyclegan import generator
from utils.utils import (cvtColor, postprocess_output, preprocess_input,
                         resize_image, show_config)


class CYCLEGAN(object):
    _defaults = {
        #-----------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #-----------------------------------------------#
        "model_path"        : 'C:/Users/15504\Documents/xwechat_files/wxid_27ngzobagsir22_5e0d\msg/file/2025-04/cyclegan-pytorch(1)/cyclegan-pytorch/model_data/Generator_B2A_horse2zebra.pth',
        #-----------------------------------------------#
        #   输入图像大小的设置
        #-----------------------------------------------#
        "input_shape"       : [256, 256],
        #-------------------------------#
        #   是否进行不失真的resize
        #-------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : False,
    }

    #---------------------------------------------------#
    #   初始化CYCLEGAN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
            self._defaults[name] = value 
        self.generate()
        
        show_config(**self._defaults)

    def load_mapped_weights(self):
        """加载权重并自动转换键名映射"""
        # 原始权重加载

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        
        # 键名映射规则（根据你的报错信息定制）
        key_mapping = {
            # 第一层映射
            "model.1.weight": "stage_1.1.weight",
            "model.1.bias": "stage_1.1.bias",
            
            # 第二层映射
            "model.4.weight": "stage_2.0.weight",
            "model.4.bias": "stage_2.0.bias",
            
            # 第三层映射
            "model.7.weight": "stage_3.0.weight",
            "model.7.bias": "stage_3.0.bias",
            
            # 残差块部分（示例2个，实际需要按报错信息补全）
            "model.10.conv_block.1.weight": "stage_4.0.conv_block.1.weight",
            "model.10.conv_block.1.bias": "stage_4.0.conv_block.1.bias",
            "model.10.conv_block.5.weight": "stage_4.0.conv_block.5.weight",
            "model.10.conv_block.5.bias": "stage_4.0.conv_block.5.bias",
            
            # 上采样部分
            "model.19.weight": "up_stage_1.0.weight",
            "model.19.bias": "up_stage_1.0.bias",
            "model.22.weight": "up_stage_2.0.weight",
            "model.22.bias": "up_stage_2.0.bias",
            
            # 输出头
            "model.26.weight": "head.1.weight",
            "model.26.bias": "head.1.bias"
        }
        
        # 自动补全所有conv_block的映射（4.0-4.8）
        for i in range(9):  # 对应stage_4.0到stage_4.8
            old_prefix = f"model.{10+i}.conv_block"
            new_prefix = f"stage_4.{i}.conv_block"
            key_mapping.update({
                f"{old_prefix}.1.weight": f"{new_prefix}.1.weight",
                f"{old_prefix}.1.bias": f"{new_prefix}.1.bias",
                f"{old_prefix}.5.weight": f"{new_prefix}.5.weight",
                f"{old_prefix}.5.bias": f"{new_prefix}.5.bias"
            })
        
        # 执行键名转换
        new_state_dict = {}
        for old_key, value in state_dict.items():
            if old_key in key_mapping:
                new_key = key_mapping[old_key]
                new_state_dict[new_key] = value
            else:
                print(f"警告: 未映射的键 {old_key}")
        
        # 加载转换后的权重
        missing_keys, unexpected_keys = self.net.load_state_dict(new_state_dict, strict=False)
        
        # 打印调试信息
        if missing_keys:
            print("\n以下键未加载:")
            for k in missing_keys:
                print(f"- {k}")
        
        if unexpected_keys:
            print("\n以下权重未使用:")
            for k in unexpected_keys:
                print(f"- {k}")

    def generate(self):
        #----------------------------------------#
        #   创建GAN模型
        #----------------------------------------#
        self.net    = generator().eval()

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net.to(device)
        self.load_mapped_weights()
        self.net    = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        # if self.cuda:
            # self.net = nn.DataParallel(self.net)
            # self.net = self.net.cuda()

    #---------------------------------------------------#
    #   生成1x1的图片
    #---------------------------------------------------#
    def detect_image(self, image):
    #---------------------------------------------------------#
      #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
      #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
      #---------------------------------------------------------#
      image = cvtColor(image)
      
      #---------------------------------------------------#
      #   获得高宽
      #---------------------------------------------------#
      orininal_h = np.array(image).shape[0]
      orininal_w = np.array(image).shape[1]
      
      #---------------------------------------------------------#
      #   给图像增加灰条，实现不失真的resize
      #   也可以直接resize进行识别
      #---------------------------------------------------------#
      image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
      
      #---------------------------------------------------------#
      #   添加上batch_size维度
      #---------------------------------------------------------#
      image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
      
      with torch.no_grad():
          images = torch.from_numpy(image_data)
          if self.cuda:
              images = images.cuda()
              
          #---------------------------------------------------#
          #   图片传入网络进行预测
          #---------------------------------------------------#
          pr = self.net(images)[0]
          #---------------------------------------------------#
          #   转为numpy
          #---------------------------------------------------#
          pr = pr.permute(1, 2, 0).cpu().numpy()
          
          #--------------------------------------#
          #   将灰条部分截取掉
          #--------------------------------------#
          if nw is not None:
              pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh),
                      int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
              
          #---------------------------------------------------#
          #   进行图片的resize
          #---------------------------------------------------#
          pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
          
      # 修改后的输出处理
      image = postprocess_output(pr)
      image = np.uint8(image)
      
      # 返回numpy数组格式（兼容Colab显示）
      return image