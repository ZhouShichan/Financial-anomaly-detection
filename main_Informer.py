import torch

from models.Informer import Model
from utils.dgraphfin import load_data, get_adj_nodes


# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = load_data('./datasets/632d74d4e2843a53167ee9a1-momodel/', 'DGraph', force_to_symmetric=True)

num_nodes = 8  # 节点数量，占比超过 99%

model_params = {
    "enc_in": 20,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 3,
    "d_ff": 2048,
    "factor": 5,
    "dropout": 0.05,
    "activation": 'gelu',
}

model = Model(
    task_name='classification',
    label_len=num_nodes,
    output_attention=False,
    num_class=2,
    seq_len=num_nodes,
    **model_params
).to(device)
model_desc = f'Informer-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'
model_save_path = f'results/model-{model_desc}.pt'
model.load_state_dict(torch.load(model_save_path, map_location=device))


def predict(data, node_id):
    x, x_mark, _, _ = get_adj_nodes(data, node_id, num_nodes)
    x, x_mark = x.to(device).unsqueeze(0), x_mark.to(device).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        pred = model(x, x_mark).exp()
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
