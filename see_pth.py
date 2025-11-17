import argparse
import torch
from models.model import load_model
import pickle
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import numpy as np
import os


def get_normalized_laplacian(A, laplacian_type='sym'):
    """
    计算归一化图拉普拉斯矩阵

    Args:
        A: 邻接矩阵 (torch.Tensor 或 numpy.ndarray), shape [V, V]
        laplacian_type: 拉普拉斯类型, 'sym' 或 'rw'
            'sym': 对称归一化拉普拉斯 L = I - D^(-1/2) A D^(-1/2)
            'rw': 随机游走归一化拉普拉斯 L = I - D^(-1) A

    Returns:
        L: 归一化图拉普拉斯矩阵, shape [V, V]
    """

    # 处理不同类型的输入
    if isinstance(A, csr_matrix):
        # 如果是 SciPy 稀疏矩阵，先转换为稠密矩阵再转为 PyTorch Tensor
        A_dense = A.toarray()
        A = torch.from_numpy(A_dense).float()
    elif isinstance(A, np.ndarray):
        # 如果是 numpy 数组，直接转为 PyTorch Tensor
        A = torch.from_numpy(A).float()
    # 如果已经是 torch.Tensor，保持不变

    V = A.shape[0]

    # 计算度矩阵 D
    degrees = torch.sum(A, dim=1)  # 每个节点的度
    D = torch.diag(degrees)        # 度矩阵

    # 计算单位矩阵 I
    I = torch.eye(V)

    if laplacian_type == 'sym':
        # 对称归一化拉普拉斯: L = I - D^(-1/2) A D^(-1/2)

        # 处理孤立节点（度为0的节点）
        degrees[degrees == 0] = 1  # 避免除零

        # 计算 D^(-1/2)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees))

        # 计算 L
        L = I - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)

    elif laplacian_type == 'rw':
        # 随机游走归一化拉普拉斯: L = I - D^(-1) A

        # 处理孤立节点
        degrees[degrees == 0] = 1  # 避免除零

        # 计算 D^(-1)
        D_inv = torch.diag(1.0 / degrees)

        # 计算 L
        L = I - torch.mm(D_inv, A)

    else:
        raise ValueError("laplacian_type 必须是 'sym' 或 'rw'")

    return L

def csr_to_scaled_laplacian(adj_csr, lambda_max=None):
    """
    将CSR格式的邻接矩阵转换为缩放拉普拉斯矩阵
    
    参数:
    adj_csr: scipy.sparse.csr_matrix, CSR格式的邻接矩阵
    lambda_max: float, 可选的最大特征值，如果为None则自动计算
    
    返回:
    scaled_L_csr: CSR格式的缩放拉普拉斯矩阵
    """
    
    # 获取节点数量
    n_nodes = adj_csr.shape[0]
    
    # 计算度矩阵 D
    degrees = np.array(adj_csr.sum(axis=1)).flatten()
    
    # 创建度矩阵的CSR格式
    D_csr = sp.diags(degrees, format='csr')
    
    # 计算拉普拉斯矩阵 L = D - A
    L_csr = D_csr - adj_csr
    
    # 如果未提供最大特征值，则计算L的最大特征值
    if lambda_max is None:
        # 使用稀疏特征值计算，只计算最大特征值
        try:
            eigenvalues, _ = eigs(L_csr, k=1, which='LM')
            lambda_max = np.real(eigenvalues[0])
        except:
            # 如果计算失败，使用近似值 2.0
            lambda_max = 2.0
    
    # 计算缩放拉普拉斯矩阵: tilde{L} = 2L/lambda_max - I
    # 首先计算 2L/lambda_max
    scaled_L_csr = (2.0 / lambda_max) * L_csr
    
    # 减去单位矩阵 I
    I_csr = sp.identity(n_nodes, format='csr')
    scaled_L_csr = scaled_L_csr - I_csr
    
    return scaled_L_csr

def csr_to_symmetric_scaled_laplacian(adj_csr, lambda_max=None):
    """
    将CSR格式的邻接矩阵转换为对称归一化的缩放拉普拉斯矩阵
    
    参数:
    adj_csr: scipy.sparse.csr_matrix, CSR格式的邻接矩阵
    lambda_max: float, 可选的最大特征值，如果为None则自动计算
    
    返回:
    scaled_L_csr: CSR格式的对称归一化缩放拉普拉斯矩阵
    """
    
    # 获取节点数量
    n_nodes = adj_csr.shape[0]
    
    # 计算度矩阵 D
    degrees = np.array(adj_csr.sum(axis=1)).flatten()
    
    # 避免除零错误，将度为0的节点度设为1
    degrees[degrees == 0] = 1
    
    # 计算 D^{-1/2}
    D_sqrt_inv = sp.diags(1.0 / np.sqrt(degrees), format='csr')
    
    # 计算对称归一化拉普拉斯矩阵: L_sym = I - D^{-1/2} A D^{-1/2}
    I_csr = sp.identity(n_nodes, format='csr')
    L_sym = I_csr - D_sqrt_inv @ adj_csr @ D_sqrt_inv
    
    # 如果未提供最大特征值，则计算L_sym的最大特征值
    if lambda_max is None:
        try:
            eigenvalues, _ = eigs(L_sym, k=1, which='LM')
            lambda_max = np.real(eigenvalues[0])
        except:
            # 对称归一化拉普拉斯矩阵的特征值范围通常是[0,2]
            lambda_max = 2.0
    
    # 计算缩放拉普拉斯矩阵: tilde{L} = 2L_sym/lambda_max - I
    scaled_L_csr = (2.0 / lambda_max) * L_sym - I_csr
    
    return scaled_L_csr

# 使用示例
if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", type=str, default='misc/model/config.yaml')
    # parser.add_argument("--model", type=str, default='output/model/exp/15.pth')
    # parser.add_argument("--search", type=str, default='')
    # opt = parser.parse_args()

    # state = torch.load(opt.model, map_location='cpu')
    # # 打印检查点的键
    # print("检查点中的键:")
    # for key in state.keys():
    #     if opt.search in key:
    #         print(key)

    # network = load_model(opt.cfg)
    # # 打印模型期望的键
    # print("\n模型期望的键:")
    # for key in network.state_dict().keys():
    #     if opt.search in key:
    #         print(key)
    
    # directory='misc/mesh_right.pkl'
    # with open(directory, "rb") as file:
    #     left_graph_dict = pickle.load(file)

    # path = 'misc/graph_left.pkl'
    # with open(path, "rb") as file:
    #     left_graph_dict = pickle.load(file)
    # mesh_adj=left_graph_dict['mesh_adj']
    # symmetric_scaled_laplacian = csr_to_symmetric_scaled_laplacian(mesh_adj)
    # dict_left = {'mesh_L': symmetric_scaled_laplacian,'mesh_adj':mesh_adj}
    # directory='misc/mesh_left.pkl'
    # if directory and not os.path.exists(directory):
    #     os.makedirs(directory)
    #     print(f"创建目录: {directory}")
    # with open(directory, "wb") as file:
    #     pickle.dump(dict_left, file)
        
        
    # path = 'misc/graph_right.pkl'
    # with open(path, "rb") as file:
    #     right_graph_dict = pickle.load(file)
    # mesh_adj=right_graph_dict['mesh_adj']
    # symmetric_scaled_laplacian = csr_to_symmetric_scaled_laplacian(mesh_adj)
    # dict_right = {'mesh_L': symmetric_scaled_laplacian,'mesh_adj':mesh_adj}
    
    # directory='misc/mesh_right.pkl'
    # if directory and not os.path.exists(directory):
    #     os.makedirs(directory)
    #     print(f"创建目录: {directory}")
    # with open(directory, "wb") as file:
    #     pickle.dump(dict_right, file)
    
        
    directory='./misc/mano/MANO_LEFT.pkl'
    manoData = pickle.load(open(directory, 'rb'), encoding='latin1')
    
    print()
        
