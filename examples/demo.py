import requests
import numpy as np
import pandas as pd
import json

# 1. 数据集管理
# 查看可用数据集
datasets = requests.get('http://localhost:5000/datasets').json()

# 获取内置数据集
iris_data = requests.get('http://localhost:5000/datasets/built_in_iris').json()

# 上传新数据集
new_dataset = {
    'name': 'my_dataset',
    'description': '自定义数据集',
    'X': [[1, 2], [3, 4], [5, 6]],
    'y': [0, 1, 0],
    'feature_names': ['feature1', 'feature2'],
    'target_names': ['class0', 'class1']
}
response = requests.post('http://localhost:5000/datasets', json=new_dataset)
dataset_id = response.json()['dataset_id']

data = requests.get(f'http://localhost:5000/datasets/{dataset_id}').json()
print(data)
X = data['X']
y = data['y']
meta_data = data['meta_data']

# 2. 使用数据集训练模型
response = requests.post('http://localhost:5000/train', json={
    'dataset_id': dataset_id,  # 使用已上传的数据集
    'model_type': 'logistic',
    'params': {'max_iter': 1000}
})
model_id = response.json()['model_id']

# 3. 预测
predictions = requests.post('http://localhost:5000/predict', json={
    'X': X,
    'model_id': model_id
}).json()

# 4. 评估模型
evaluation = requests.post('http://localhost:5000/evaluate', json={
    'X': X,
    'y': y,
    'model_id': model_id,
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
}).json()

# 5. 可视化
visualization = requests.post('http://localhost:5000/visualize', json={
    'X': X,
    'y': y,
    'model_id': model_id,
    'plot_type': 'scatter'
}).json()

# 6. 生成综合报告
report = requests.post('http://localhost:5000/generate_report', json={
    'model_id': model_id,
    'dataset_info': {
        'name': 'iris',
        'description': meta_data['description'],
        'feature_names': meta_data['feature_names'],
        'target_names': meta_data['target_names'],
        'shape': list(meta_data['shape'].values())
    },
    'training_info': {
        'model_type': 'logistic',
        'parameters': {},
        'training_time': '2023-XX-XX...',
        'convergence_info': '模型收敛信息'
    },
    'prediction_results': {
        'sample_predictions': predictions['predictions'][:5],
        'prediction_distribution': {'0': 30, '1': 40, '2': 30}
    },
    'evaluation_results': evaluation,
    'visualization_results': {
        'plots': [visualization['image']]
    }
}).json()
print(json.dumps(report, indent=4))

# 清理（可选）
# 删除数据集
requests.delete(f'http://localhost:5000/datasets/{dataset_id}')
# 删除模型
requests.delete(f'http://localhost:5000/models/{model_id}')