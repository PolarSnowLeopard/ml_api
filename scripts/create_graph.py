from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# 加载.env文件
load_dotenv()

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """完全清空数据库，包括所有节点、关系、索引和约束"""
        with self.driver.session() as session:
            try:
                # 删除所有约束
                print("正在删除所有约束...")
                # Neo4j 3.5.x 使用 schema 命名空间
                constraints = session.run("CALL db.constraints").data()
                for constraint in constraints:
                    # 3.5.x 版本中约束信息在 description 字段
                    description = constraint.get('description', '')
                    if description:
                        try:
                            # 尝试提取约束信息并删除
                            session.run(f"DROP {description}")
                            print(f"已删除约束: {description}")
                        except Exception as e:
                            print(f"删除约束时出错: {str(e)}")
                
                # 删除所有索引
                print("正在删除所有索引...")
                indexes = session.run("CALL db.indexes").data()
                for index in indexes:
                    # 3.5.x 版本中索引信息在 description 字段
                    description = index.get('description', '')
                    if description and 'INDEX' in description:
                        try:
                            # 尝试提取索引信息并删除
                            session.run(f"DROP {description}")
                            print(f"已删除索引: {description}")
                        except Exception as e:
                            print(f"删除索引时出错: {str(e)}")
                
                # 删除所有节点和关系
                print("正在删除所有节点和关系...")
                session.run("MATCH (n) DETACH DELETE n")
                
                print("数据库已完全清空")
            except Exception as e:
                print(f"清空数据库时出错：{str(e)}")
                # 如果高级清理失败，尝试基础清理
                print("尝试基础清理...")
                # 删除所有节点和关系
                session.run("MATCH (n) DETACH DELETE n")
                print("已完成基础清理")

    def create_graph(self):
        """创建包含API、类别和元应用的图数据库"""
        with self.driver.session() as session:
            # 创建API功能类别节点
            print("创建功能类别节点...")
            session.run("""
                CREATE (n1:Category {name: '数据准备'})
                CREATE (n2:Category {name: '模型管理'})
                CREATE (n3:Category {name: '报告生成'})
            """)

            # 创建API节点
            print("创建API节点...")
            session.run("""
                // 数据准备API
                CREATE (n1:API {name: 'GET /datasets', method: 'GET', path: '/datasets'})
                CREATE (n2:API {name: 'POST /datasets', method: 'POST', path: '/datasets'})
                CREATE (n3:API {name: 'GET /datasets/id', method: 'GET', path: '/datasets/id'})
                CREATE (n4:API {name: 'DELETE /datasets/id', method: 'DELETE', path: '/datasets/id'})
                
                // 模型管理API
                CREATE (n5:API {name: 'GET /models', method: 'GET', path: '/models'})
                CREATE (n6:API {name: 'GET /models/id', method: 'GET', path: '/models/id'})
                CREATE (n7:API {name: 'DELETE /models/id', method: 'DELETE', path: '/models/id'})
                CREATE (n8:API {name: 'POST /train', method: 'POST', path: '/train'})
                CREATE (n9:API {name: 'POST /predict', method: 'POST', path: '/predict'})
                CREATE (n10:API {name: 'POST /evaluate', method: 'POST', path: '/evaluate'})
                CREATE (n11:API {name: 'POST /visualize', method: 'POST', path: '/visualize'})
                
                // 报告生成API
                CREATE (n12:API {name: 'POST /generate_report', method: 'POST', path: '/generate_report'})
            """)

            # 创建元应用节点
            print("创建元应用节点...")
            session.run("""
                CREATE (n1:MetaApp {name: '模型评估元应用', 
                    description: '用于评估模型性能的基础应用'})
                CREATE (n2:MetaApp {name: '模型应用元应用', 
                    description: '包含数据集选择、模型选择、评估、预测和数据集上传的完整应用流程'})
                CREATE (n3:MetaApp {name: '模型构建-报告生成元应用', 
                    description: '包含除删除操作外的所有功能的完整流程'})
                CREATE (n4:MetaApp {name: '管理元应用', 
                    description: '用于管理数据集和模型的基础操作'})
            """)

            # 创建API之间的依赖关系
            print("创建API依赖关系...")
            
            # 数据集依赖关系
            session.run("""
                MATCH (post:API {path: '/datasets', method: 'POST'}), 
                      (get:API {path: '/datasets/id', method: 'GET'})
                CREATE (post)-[:REQUIRED_BY]->(get)
            """)
            
            session.run("""
                MATCH (get_dataset:API {path: '/datasets/id', method: 'GET'}),
                      (target:API)
                WHERE target.path IN ['/train', '/evaluate', '/visualize'] 
                AND target.method = 'POST'
                CREATE (get_dataset)-[:REQUIRED_BY]->(target)
            """)
            
            session.run("""
                MATCH (get_all:API {path: '/datasets', method: 'GET'}),
                      (get_one:API {path: '/datasets/id', method: 'GET'})
                CREATE (get_all)-[:REQUIRED_BY]->(get_one)
            """)
            
            session.run("""
                MATCH (list_api:API {path: '/datasets'}),
                      (delete_api:API {path: '/datasets/id', method: 'DELETE'})
                CREATE (list_api)-[:REQUIRED_BY]->(delete_api)
            """)
            
            # 训练依赖关系
            session.run("""
                MATCH (train:API {path: '/train'}),
                      (model:API {path: '/models/id'})
                WHERE model.method IN ['GET', 'DELETE']
                CREATE (train)-[:REQUIRED_BY]->(model)
            """)
            
            # 模型使用依赖关系
            session.run("""
                MATCH (get_model:API {path: '/models/id', method: 'GET'}),
                      (target:API)
                WHERE target.path IN ['/predict', '/evaluate', '/visualize']
                AND target.method = 'POST'
                CREATE (get_model)-[:REQUIRED_BY]->(target)
            """)
            
            session.run("""
                MATCH (list_models:API {path: '/models', method: 'GET'}),
                      (target:API {path: '/models/id'})
                WHERE target.method IN ['GET', 'DELETE']
                CREATE (list_models)-[:REQUIRED_BY]->(target)
            """)
            
            # 报告生成依赖关系
            session.run("""
                MATCH (source:API), (report:API {path: '/generate_report'})
                WHERE source.path IN ['/predict', '/evaluate', '/visualize']
                AND source.method = 'POST'
                CREATE (source)-[:REQUIRED_BY]->(report)
            """)

            # 创建API和功能类别的从属关系
            print("创建API类别关系...")
            session.run("""
                // 数据准备类别
                MATCH (a:API), (c:Category)
                WHERE a.path CONTAINS 'datasets' AND c.name = '数据准备'
                CREATE (a)-[:BELONGS_TO]->(c)
            """)
            
            session.run("""
                // 模型管理类别
                MATCH (a:API), (c:Category)
                WHERE (a.path CONTAINS 'models' OR 
                      a.path IN ['/train', '/predict', '/evaluate', '/visualize'])
                AND c.name = '模型管理'
                CREATE (a)-[:BELONGS_TO]->(c)
            """)
            
            session.run("""
                // 报告生成类别
                MATCH (a:API), (c:Category)
                WHERE a.path = '/generate_report' AND c.name = '报告生成'
                CREATE (a)-[:BELONGS_TO]->(c)
            """)

            # 创建元应用和API的组合关系
            print("创建元应用关系...")
            
            # 1. 模型评估元应用
            session.run("""
                MATCH (m:MetaApp), (a:API)
                WHERE m.name = '模型评估元应用' 
                AND a.name IN ['GET /models/id', 'GET /datasets/id', 'POST /evaluate']
                CREATE (m)-[:USES]->(a)
            """)
            
            # 2. 模型应用元应用
            session.run("""
                MATCH (m:MetaApp), (a:API)
                WHERE m.name = '模型应用元应用' 
                AND a.name IN [
                    'GET /datasets/id',
                    'GET /models/id',
                    'POST /evaluate',
                    'POST /predict',
                    'POST /datasets'
                ]
                CREATE (m)-[:USES]->(a)
            """)
            
            # 3. 模型构建-报告生成元应用
            session.run("""
                MATCH (m:MetaApp), (a:API)
                WHERE m.name = '模型构建-报告生成元应用' 
                AND a.name IN [
                    'GET /datasets',
                    'POST /datasets',
                    'GET /datasets/id',
                    'GET /models',
                    'GET /models/id',
                    'POST /train',
                    'POST /predict',
                    'POST /evaluate',
                    'POST /visualize',
                    'POST /generate_report'
                ]
                CREATE (m)-[:USES]->(a)
            """)
            
            # 4. 管理元应用
            session.run("""
                MATCH (m:MetaApp), (a:API)
                WHERE m.name = '管理元应用' 
                AND a.name IN [
                    'GET /datasets',
                    'GET /models',
                    'GET /datasets/id',
                    'GET /models/id',
                    'DELETE /models/id',
                    'DELETE /datasets/id'
                ]
                CREATE (m)-[:USES]->(a)
            """)
            
            print("图数据库创建完成！")

if __name__ == "__main__":
    # 从环境变量获取Neo4j配置
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        print("错误：请确保在.env文件中设置了所有必需的Neo4j配置")
        exit(1)

    try:
        # 创建连接
        conn = Neo4jConnection(uri, user, password)
        
        # 先清空数据库
        print("正在清空数据库...")
        conn.clear_database()
        
        # 创建新的图
        print("正在创建新的图...")
        conn.create_graph()
        print("图数据库创建成功！")
    except Exception as e:
        print(f"操作数据库时出错：{str(e)}")
    finally:
        conn.close() 