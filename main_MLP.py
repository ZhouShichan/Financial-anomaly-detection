import torch

from models.MLP import Model
from utils.dgraphfin import load_data, get_adj_nodes


# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# 加载数据
data = load_data('./datasets/632d74d4e2843a53167ee9a1-momodel/', 'DGraph', force_to_symmetric=True)
data = data.to(device)

model_params = {
    "hidden_channels": 16,
    "num_layers": 2,
    "dropout": 0.1,
}

model = Model(
    in_channels=20,
    out_channels=2,
    batchnorm=False,
    ** model_params
).to(device)
model_desc = f'MLP-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'
model_save_path = f'results/model-{model_desc}.pt'
model.load_state_dict(torch.load(model_save_path, map_location=device))


def predict(data, node_id):
    if isinstance(node_id, int):
        node_id = torch.tensor([node_id])
    if node_id.dim() == 0:
        node_id = node_id.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        pred = model(data.x[node_id]).exp()
    return pred.squeeze(0)


if __name__ == '__main__':
    dic = {0: "正常用户", 1: "欺诈用户"}
    node_idx = 0
    y_pred = predict(data, node_idx)
    print(y_pred)
    print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')

    node_idx = 1
    y_pred = predict(data, node_idx)
    print(y_pred)
    print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')
