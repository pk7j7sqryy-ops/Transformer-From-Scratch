'''
    项目脚本运行主入口
'''
from sys import argv

'''
# 不用 main.py，要记 5 条命令：
python proprocess/preprocess.py        # 预处理
python runner/training.py              # 训练
python runner/evaluate.py             # 评估
python runner/predict.py              # 翻译
python web/app.py                     # 启动服务

# 有 main.py，只记 1 条命令 + 参数：
python main.py preprocess
python main.py train
python main.py evaluate
python main.py predict
python main.py serve

python main.py serve
         ↑        ↑
       argv[0]  argv[1]
'''

if __name__ == '__main__':
    # print(argv)

    # 获取第二个参数，要执行的操作
    action = argv[1]

    match action:
        case 'preprocess':
            from proprocess.preprocess import preprocess
            preprocess()
        case 'train':
            from runner.training import train
            train()
        case 'evaluate':
            from runner.evaluate import run_evaluation
            run_evaluation()
        case 'predict':
            from runner.predict import predict
            predict()
        case 'serve':
            from web.app import run_app
            run_app()