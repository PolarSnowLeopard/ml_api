import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_iris, load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import json
from datetime import datetime
import os
import joblib
import hashlib

app = Flask(__name__)
api = Api(app, version='1.0', title='机器学习分类算法 API',
          description='提供机器学习分类算法训练、预测、评估和可视化功能的REST API')

# 创建存储目录
MODELS_DIR = 'saved_models'
DATASETS_DIR = 'saved_datasets'
for directory in [MODELS_DIR, DATASETS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 存储模型和数据集的字典（内存缓存）
models = {}
datasets = {}

def save_dataset(dataset_id, data, meta_data):
    """保存数据集到文件系统"""
    try:
        # 保存数据
        data_path = os.path.join(DATASETS_DIR, dataset_id)
        joblib.dump(data, data_path)
        
        # 保存元数据
        meta_path = os.path.join(DATASETS_DIR, f"{dataset_id}.meta")
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f)
        return True
    except Exception as e:
        print(f"保存数据集 {dataset_id} 时出错: {str(e)}")
        return False

def load_saved_datasets():
    """加载所有保存的数据集到内存"""
    # 首先加载内置数据集
    built_in_datasets = {
        'iris': load_iris(),
        'breast_cancer': load_breast_cancer()
    }
    
    for name, data in built_in_datasets.items():
        dataset_id = f"built_in_{name}"
        X, y = data.data, data.target
        meta_data = {
            'name': name,
            'description': data.DESCR,
            'feature_names': list(data.feature_names),
            'target_names': list(data.target_names),
            'shape': {'samples': X.shape[0], 'features': X.shape[1]},
            'type': 'built_in',
            'creation_time': datetime.now().isoformat()
        }
        datasets[dataset_id] = {
            'X': X,
            'y': y,
            'meta_data': meta_data
        }
        # 保存内置数据集
        save_dataset(dataset_id, {'X': X, 'y': y}, meta_data)
    
    # 然后加载用户上传的数据集
    for dataset_id in os.listdir(DATASETS_DIR):
        if dataset_id.endswith('.meta'):
            continue
        if dataset_id.startswith('built_in_'):
            continue
        data_path = os.path.join(DATASETS_DIR, dataset_id)
        meta_path = os.path.join(DATASETS_DIR, f"{dataset_id}.meta")
        if os.path.exists(data_path) and os.path.exists(meta_path):
            try:
                # 加载数据
                data = joblib.load(data_path)
                # 加载元数据
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                # 存储到内存
                datasets[dataset_id] = {
                    'X': data['X'],
                    'y': data['y'],
                    'meta_data': meta_data
                }
            except Exception as e:
                print(f"加载数据集 {dataset_id} 时出错: {str(e)}")

def generate_dataset_id(name, user_id='default'):
    """生成唯一的数据集ID"""
    unique_string = f"{name}_{user_id}_{datetime.now().isoformat()}"
    return hashlib.md5(unique_string.encode()).hexdigest()[:8]

def save_model(model_id, model_data, meta_data):
    """保存模型和元数据到文件系统"""
    try:
        # 保存模型对象
        model_path = os.path.join(MODELS_DIR, model_id)
        joblib.dump({
            'model': model_data['model'],
            'scaler': model_data['scaler']
        }, model_path)
        
        # 保存元数据
        meta_path = os.path.join(MODELS_DIR, f"{model_id}.meta")
        with open(meta_path, 'w') as f:
            json.dump({
                'creation_time': meta_data['creation_time'],
                'model_type': meta_data['model_type'],
                'params': meta_data['params'],
                'dataset_info': meta_data['dataset_info']
            }, f)
        return True
    except Exception as e:
        print(f"保存模型 {model_id} 时出错: {str(e)}")
        return False

def load_saved_models():
    """加载所有保存的模型到内存"""
    for model_id in os.listdir(MODELS_DIR):
        if model_id.endswith('.meta'):
            continue
        model_path = os.path.join(MODELS_DIR, model_id)
        meta_path = os.path.join(MODELS_DIR, f"{model_id}.meta")
        if os.path.exists(model_path) and os.path.exists(meta_path):
            try:
                # 加载模型对象
                model_data = joblib.load(model_path)
                # 加载元数据
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                # 将模型和元数据存储到内存中
                models[model_id] = {
                    'model': model_data['model'],
                    'scaler': model_data['scaler'],
                    'creation_time': meta_data['creation_time'],
                    'model_type': meta_data['model_type'],
                    'params': meta_data['params'],
                    'dataset_info': meta_data['dataset_info']
                }
            except Exception as e:
                print(f"加载模型 {model_id} 时出错: {str(e)}")

def generate_model_id(model_type, dataset_info):
    """生成唯一的模型ID"""
    unique_string = f"{model_type}_{dataset_info['name']}_{datetime.now().isoformat()}"
    return hashlib.md5(unique_string.encode()).hexdigest()[:8]

# 在启动时加载已保存的数据集和模型
load_saved_datasets()
load_saved_models()

# 定义数据集上传模型
dataset_upload_model = api.model('DatasetUpload', {
    'name': fields.String(required=True, description='数据集名称'),
    'description': fields.String(required=True, description='数据集描述'),
    'X': fields.List(fields.List(fields.Float), required=True, description='特征矩阵'),
    'y': fields.List(fields.Integer, required=True, description='目标变量'),
    'feature_names': fields.List(fields.String, required=True, description='特征名列表'),
    'target_names': fields.List(fields.String, required=True, description='目标类别名列表')
})

# 定义API模型
train_model = api.model('TrainModel', {
    'dataset_id': fields.String(description='数据集ID（如果使用已存储的数据集）'),
    'X': fields.List(fields.List(fields.Float), description='特征矩阵（如果不使用已存储的数据集）'),
    'y': fields.List(fields.Integer, description='目标变量（如果不使用已存储的数据集）'),
    'model_type': fields.String(required=True, enum=['logistic', 'svm', 'decision_tree'], description='模型类型'),
    'params': fields.Raw(description='模型参数（可选）')
})

predict_model = api.model('PredictModel', {
    'X': fields.List(fields.List(fields.Float), required=True, description='特征矩阵'),
    'model_id': fields.String(required=True, description='模型ID')
})

evaluate_model = api.model('EvaluateModel', {
    'X': fields.List(fields.List(fields.Float), required=True, description='特征矩阵'),
    'y': fields.List(fields.Integer, required=True, description='真实标签'),
    'model_id': fields.String(required=True, description='模型ID'),
    'metrics': fields.List(fields.String, required=True, description='评估指标列表')
})

visualize_model = api.model('VisualizeModel', {
    'X': fields.List(fields.List(fields.Float), required=True, description='特征矩阵'),
    'y': fields.List(fields.Integer, required=True, description='目标变量'),
    'model_id': fields.String(required=True, description='模型ID'),
    'plot_type': fields.String(required=True, enum=['scatter', 'decision_boundary'], description='可视化类型')
})

# 定义报告生成API的请求和响应模型
dataset_info_model = api.model('DatasetInfo', {
    'name': fields.String(required=True, description='数据集名称'),
    'description': fields.String(required=True, description='数据集描述'),
    'feature_names': fields.List(fields.String, required=True, description='特征名列表'),
    'target_names': fields.List(fields.String, required=True, description='目标类别名列表'),
    'shape': fields.List(fields.Integer, required=True, description='数据集形状 [样本数, 特征数]')
})

training_info_model = api.model('TrainingInfo', {
    'model_type': fields.String(required=True, description='模型类型'),
    'parameters': fields.Raw(description='模型参数'),
    'training_time': fields.String(required=True, description='训练时间'),
    'convergence_info': fields.String(description='模型收敛信息')
})

prediction_results_model = api.model('PredictionResults', {
    'sample_predictions': fields.List(fields.Integer, required=True, description='预测样本结果'),
    'prediction_distribution': fields.Raw(required=True, description='预测结果分布')
})

evaluation_results_model = api.model('EvaluationResults', {
    'accuracy': fields.Float(description='准确率'),
    'precision': fields.Float(description='精确率'),
    'recall': fields.Float(description='召回率'),
    'f1': fields.Float(description='F1分数'),
    'confusion_matrix': fields.List(fields.List(fields.Integer), description='混淆矩阵')
})

visualization_results_model = api.model('VisualizationResults', {
    'plots': fields.List(fields.String, required=True, description='Base64编码的图像列表')
})

generate_report_model = api.model('GenerateReport', {
    'model_id': fields.String(required=True, description='模型ID'),
    'dataset_info': fields.Nested(dataset_info_model, required=True, description='数据集信息'),
    'training_info': fields.Nested(training_info_model, required=True, description='训练信息'),
    'prediction_results': fields.Nested(prediction_results_model, required=True, description='预测结果'),
    'evaluation_results': fields.Nested(evaluation_results_model, required=True, description='评估结果'),
    'visualization_results': fields.Nested(visualization_results_model, required=True, description='可视化结果')
})

report_response_model = api.model('ReportResponse', {
    'report_time': fields.String(required=True, description='报告生成时间'),
    'report_sections': fields.Raw(required=True, description='报告各个部分的详细内容'),
    'conclusion': fields.String(required=True, description='综合分析结论')
})

# 数据集API
@api.route('/datasets')
class DatasetList(Resource):
    @api.doc('获取所有数据集列表')
    def get(self):
        """获取所有可用数据集列表"""
        dataset_list = []
        for dataset_id, dataset in datasets.items():
            meta_data = dataset['meta_data']
            dataset_list.append({
                'dataset_id': dataset_id,
                'name': meta_data['name'],
                'description': meta_data['description'],
                'shape': meta_data['shape'],
                'type': meta_data.get('type', 'user_uploaded'),
                'creation_time': meta_data['creation_time']
            })
        return {'datasets': dataset_list}

    @api.expect(dataset_upload_model)
    @api.doc('上传新数据集')
    def post(self):
        """上传新数据集"""
        try:
            data = request.json
            X = np.array(data['X'])
            y = np.array(data['y'])
            
            meta_data = {
                'name': data['name'],
                'description': data['description'],
                'feature_names': data['feature_names'],
                'target_names': data['target_names'],
                'shape': {'samples': X.shape[0], 'features': X.shape[1]},
                'type': 'user_uploaded',
                'creation_time': datetime.now().isoformat()
            }
            
            dataset_id = generate_dataset_id(data['name'])
            
            # 保存到内存和文件系统
            datasets[dataset_id] = {
                'X': X,
                'y': y,
                'meta_data': meta_data
            }
            
            if not save_dataset(dataset_id, {'X': X, 'y': y}, meta_data):
                return {'error': '数据集保存失败'}, 500
            
            return {
                'message': '数据集上传成功',
                'dataset_id': dataset_id
            }
            
        except Exception as e:
            return {'error': str(e)}, 400

@api.route('/datasets/<string:dataset_id>')
class Dataset(Resource):
    @api.doc('获取数据集详情')
    def get(self, dataset_id):
        """获取指定数据集的详细信息"""
        if dataset_id not in datasets:
            return {'error': '数据集不存在'}, 404
        
        dataset = datasets[dataset_id]
        return {
            'X': dataset['X'].tolist(),
            'y': dataset['y'].tolist(),
            'meta_data': dataset['meta_data']
        }
    
    @api.doc('删除数据集')
    def delete(self, dataset_id):
        """删除指定的数据集"""
        if dataset_id not in datasets:
            return {'error': '数据集不存在'}, 404
        
        if datasets[dataset_id]['meta_data'].get('type') == 'built_in':
            return {'error': '内置数据集不能删除'}, 400
        
        try:
            # 删除文件系统中的数据集文件
            data_path = os.path.join(DATASETS_DIR, dataset_id)
            meta_path = os.path.join(DATASETS_DIR, f"{dataset_id}.meta")
            
            if os.path.exists(data_path):
                os.remove(data_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            
            # 从内存中删除数据集
            del datasets[dataset_id]
            
            return {'message': '数据集删除成功'}
        except Exception as e:
            return {'error': f'删除数据集时出错: {str(e)}'}, 500

# 训练API
@api.route('/train')
class Train(Resource):
    @api.expect(train_model)
    def post(self):
        """训练新模型"""
        try:
            data = request.json
            model_type = data['model_type']
            model_params = data.get('params', {})
            
            # 获取训练数据
            if 'dataset_id' in data:
                # 使用已存储的数据集
                if data['dataset_id'] not in datasets:
                    return {'error': '数据集不存在'}, 404
                dataset = datasets[data['dataset_id']]
                X = dataset['X']
                y = dataset['y']
                dataset_info = dataset['meta_data']
            else:
                # 使用请求中提供的数据
                if 'X' not in data or 'y' not in data:
                    return {'error': '未提供数据集ID或训练数据'}, 400
                X = np.array(data['X'])
                y = np.array(data['y'])
                dataset_info = {
                    'name': 'custom_dataset',
                    'description': '用户自定义数据集',
                    'shape': {'samples': X.shape[0], 'features': X.shape[1]},
                    'type': 'custom',
                    'creation_time': datetime.now().isoformat()
                }
            
            # 数据预处理
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 选择模型
            if model_type == 'logistic':
                model = LogisticRegression(**model_params)
            elif model_type == 'svm':
                model = SVC(**model_params)
            elif model_type == 'decision_tree':
                model = DecisionTreeClassifier(**model_params)
            else:
                return {'error': '不支持的模型类型'}, 400
            
            # 训练模型
            model.fit(X_scaled, y)
            
            # 生成模型ID并保存模型
            model_id = generate_model_id(model_type, dataset_info)
            model_data = {
                'model': model,
                'scaler': scaler
            }
            meta_data = {
                'creation_time': datetime.now().isoformat(),
                'model_type': model_type,
                'params': model_params,
                'dataset_info': dataset_info
            }
            
            # 保存到内存和文件系统
            models[model_id] = {**model_data, **meta_data}
            if not save_model(model_id, model_data, meta_data):
                return {'error': '模型保存失败'}, 500
            
            return {
                'message': '模型训练成功',
                'model_id': model_id,
                'dataset_info': dataset_info
            }
            
        except Exception as e:
            return {'error': str(e)}, 400

# 预测API
@api.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    def post(self):
        """使用模型进行预测"""
        try:
            data = request.json
            X = np.array(data['X'])
            model_id = data['model_id']
            
            if model_id not in models:
                return {'error': '模型不存在'}, 404
                
            model = models[model_id]['model']
            scaler = models[model_id]['scaler']
            
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            
            return {
                'predictions': predictions.tolist()
            }
            
        except Exception as e:
            return {'error': str(e)}, 400

# 评估API
@api.route('/evaluate')
class Evaluate(Resource):
    @api.expect(evaluate_model)
    def post(self):
        """评估模型性能"""
        try:
            data = request.json
            X = np.array(data['X'])
            y = np.array(data['y'])
            model_id = data['model_id']
            metrics = data['metrics']
            
            if model_id not in models:
                return {'error': '模型不存在'}, 404
            
            model = models[model_id]['model']
            scaler = models[model_id]['scaler']
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            
            results = {}
            for metric in metrics:
                if metric == 'accuracy':
                    results['accuracy'] = accuracy_score(y, y_pred)
                elif metric == 'precision':
                    results['precision'] = precision_score(y, y_pred, average='weighted')
                elif metric == 'recall':
                    results['recall'] = recall_score(y, y_pred, average='weighted')
                elif metric == 'f1':
                    results['f1'] = f1_score(y, y_pred, average='weighted')
                elif metric == 'confusion_matrix':
                    results['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
            
            return results
            
        except Exception as e:
            return {'error': str(e)}, 400

# 可视化API
@api.route('/visualize')
class Visualize(Resource):
    @api.expect(visualize_model)
    def post(self):
        """生成模型可视化"""
        try:
            data = request.json
            X = np.array(data['X'])
            y = np.array(data['y'])
            model_id = data['model_id']
            plot_type = data['plot_type']
            
            if model_id not in models:
                return {'error': '模型不存在'}, 404
                
            plt.figure(figsize=(10, 6))
            
            if plot_type == 'scatter':
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.title('数据分布散点图')
                
            elif plot_type == 'decision_boundary':
                if X.shape[1] != 2:
                    return {'error': '决策边界图仅支持2D特征'}, 400
                    
                model = models[model_id]['model']
                scaler = models[model_id]['scaler']
                
                X_scaled = scaler.transform(X)
                
                x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
                y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                   np.arange(y_min, y_max, 0.1))
                
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                plt.contourf(xx, yy, Z, alpha=0.4)
                plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.title('决策边界')
                
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                'image': image_base64
            }
            
        except Exception as e:
            return {'error': str(e)}, 400

# 报告生成API
@api.route('/generate_report')
class GenerateReport(Resource):
    @api.expect(generate_report_model)
    @api.marshal_with(report_response_model)
    def post(self):
        """生成综合分析报告
        
        整合数据集信息、模型信息、预测结果、评估指标和可视化结果，生成完整的分析报告。
        """
        try:
            data = request.json
            model_id = data['model_id']
            
            if model_id not in models:
                return {'error': '模型不存在'}, 404
            
            report = {
                'report_time': datetime.now().isoformat(),
                'report_sections': {
                    'dataset_analysis': {
                        'summary': self._generate_dataset_summary(data['dataset_info']),
                        'feature_importance': self._analyze_feature_importance(data['dataset_info']),
                        'data_distribution': self._analyze_data_distribution(data['dataset_info'])
                    },
                    'model_analysis': {
                        'model_type': data['training_info']['model_type'],
                        'parameters': self._analyze_parameters(data['training_info']),
                        'structure_summary': self._generate_model_summary(data['training_info'])
                    },
                    'performance_analysis': {
                        'metrics_summary': self._analyze_metrics(data['evaluation_results']),
                        'prediction_analysis': self._analyze_predictions(data['prediction_results']),
                        'visualization_summary': self._analyze_visualizations(data['visualization_results'])
                    },
                    'recommendations': {
                        'model_improvements': self._generate_model_recommendations(data),
                        'feature_suggestions': self._generate_feature_suggestions(data),
                        'parameter_tuning': self._generate_parameter_suggestions(data)
                    }
                },
                'conclusion': self._generate_conclusion(data)
            }
            
            return report
            
        except Exception as e:
            return {'error': str(e)}, 400

    def _generate_dataset_summary(self, dataset_info):
        return f"数据集'{dataset_info['name']}'包含{dataset_info['shape'][0]}个样本，{dataset_info['shape'][1]}个特征。"

    def _analyze_feature_importance(self, dataset_info):
        return [f"特征'{name}'的重要性分析" for name in dataset_info['feature_names']]

    def _analyze_data_distribution(self, dataset_info):
        return "数据分布分析结果"

    def _analyze_parameters(self, training_info):
        return f"模型使用的参数配置分析"

    def _generate_model_summary(self, training_info):
        return f"{training_info['model_type']}模型结构概述"

    def _analyze_metrics(self, evaluation_results):
        metrics_summary = []
        if 'accuracy' in evaluation_results:
            metrics_summary.append(f"准确率: {evaluation_results['accuracy']:.2%}")
        if 'precision' in evaluation_results:
            metrics_summary.append(f"精确率: {evaluation_results['precision']:.2%}")
        if 'recall' in evaluation_results:
            metrics_summary.append(f"召回率: {evaluation_results['recall']:.2%}")
        if 'f1' in evaluation_results:
            metrics_summary.append(f"F1分数: {evaluation_results['f1']:.2%}")
        return ' '.join(metrics_summary)

    def _analyze_predictions(self, prediction_results):
        return f"预测结果分布分析"

    def _analyze_visualizations(self, visualization_results):
        return f"可视化结果分析，包含{len(visualization_results['plots'])}张图表"

    def _generate_model_recommendations(self, data):
        return ["根据模型性能提出的改进建议"]

    def _generate_feature_suggestions(self, data):
        return ["特征工程优化建议"]

    def _generate_parameter_suggestions(self, data):
        return ["参数调优建议"]

    def _generate_conclusion(self, data):
        evaluation_results = data['evaluation_results']
        conclusion = []
        
        if 'accuracy' in evaluation_results:
            accuracy = evaluation_results['accuracy']
            if accuracy >= 0.9:
                conclusion.append(f"模型表现优秀，准确率达到{accuracy:.2%}")
            elif accuracy >= 0.7:
                conclusion.append(f"模型表现良好，准确率为{accuracy:.2%}")
            else:
                conclusion.append(f"模型表现有待提升，���确率为{accuracy:.2%}")
        
        conclusion.append(f"基于{data['dataset_info']['name']}数据集的分析已完成")
        conclusion.append("详细的改进建议请参考recommendations部分")
        
        return ' '.join(conclusion)

# 添加模型管理API
@api.route('/models')
class ModelList(Resource):
    def get(self):
        """获取所有已保存的模型列表"""
        model_list = []
        for model_id, model_info in models.items():
            model_list.append({
                'model_id': model_id,
                'model_type': model_info['model_type'],
                'creation_time': model_info['creation_time'],
                'dataset_info': model_info['dataset_info']
            })
        return {'models': model_list}

@api.route('/models/<string:model_id>')
class Model(Resource):
    def get(self, model_id):
        """获取指定模型的详细信息"""
        if model_id not in models:
            return {'error': '模型不存在'}, 404
        
        model_info = models[model_id]
        return {
            'model_id': model_id,
            'model_type': model_info['model_type'],
            'creation_time': model_info['creation_time'],
            'params': model_info['params'],
            'dataset_info': model_info['dataset_info']
        }
    
    def delete(self, model_id):
        """删除指定的模型"""
        if model_id not in models:
            return {'error': '模型不存在'}, 404
        
        try:
            # 删除文件系统中的模型文件
            model_path = os.path.join(MODELS_DIR, model_id)
            meta_path = os.path.join(MODELS_DIR, f"{model_id}.meta")
            
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            
            # 从内存中删除模型
            del models[model_id]
            
            return {'message': '模型删除成功'}
        except Exception as e:
            return {'error': f'删除模型时出错: {str(e)}'}, 500

if __name__ == '__main__':
    app.run(debug=True) 