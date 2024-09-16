import torch

from models.MLP import Model
from utils.dgraphfin import load_data, get_adj_nodes


# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = load_data('./datasets/632d74d4e2843a53167ee9a1-momodel/', 'DGraph', force_to_symmetric=True)

num_nodes = 8  # 节点数量，占比超过 99%

model_params = {
    "hidden_channels": 16,
    "num_layers": 2,
    "dropout": 0.1,
}

model = Model(
    in_channels=num_nodes * (20 + 1),
    out_channels=2,
    batchnorm=False,
    ** model_params
).to(device)
model_desc = f'MLP-AdjNodes_{num_nodes}-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'
model_save_path = f'results/model-{model_desc}.pt'
model.load_state_dict(torch.load(model_save_path, map_location=device))


def predict(data, node_id):
    x, x_mark, _, _ = get_adj_nodes(data, node_id, num_nodes)
    x, x_mark = x.to(device).unsqueeze(0), x_mark.to(device).unsqueeze(0).unsqueeze(-1)
    x = torch.flatten(torch.cat((x, x_mark), dim=-1), 1)
    with torch.no_grad():
        model.eval()
        pred = model(x).exp()
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
