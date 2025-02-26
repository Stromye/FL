import flwr as fl
from typing import Dict, List, Tuple
import numpy as np
from flwr.common import Metrics

# 定义指标聚合函数
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# 自定义 FedAvg 策略
class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[List[np.ndarray], Dict[str, fl.common.Scalar]]:
        # 调用父类的 aggregate_fit 方法
        aggregated_weights, metrics = super().aggregate_fit(server_round, results, failures)
      
        return aggregated_weights, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Scalar, Dict[str, fl.common.Scalar]]:
        print("检查是否请求评估")
        print(f"Evaluation results from clients: {results}")
        # 调用父类的 aggregate_evaluate 方法
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if not results:
            return aggregated_loss, {}
        # 计算加权平均准确率
        metrics_eval = weighted_average([(res.num_examples, res.metrics) for _, res in results])
        metrics.update(metrics_eval)
        return aggregated_loss, metrics

# 定义策略
strategy = CustomFedAvg(
    min_fit_clients=4,  # 至少需要 4 个客户端参与训练
    min_evaluate_clients=4,  # 至少需要 4 个客户端参与评估
    min_available_clients=4,  # 至少需要 4 个客户端可用
    fit_metrics_aggregation_fn=weighted_average,  # 训练指标聚合函数
    evaluate_metrics_aggregation_fn=weighted_average,  # 评估指标聚合函数
)

# 启动 Flower 服务器
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
