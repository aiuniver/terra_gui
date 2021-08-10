from terra_ai.data.training.train import TrainData

TRAINDATA = {
    "batch": 32,
    "epochs": 20,
    "optimizer": {
        "type": "Adam",
        "parameters": {
            "main": {"learning_rate": 0.0001},
            "extra": {
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-07,
                "amsgrad": False
            }
        }
    },
    "architecture": {
        "type": "Basic",
        "parameters": {
            "outputs": [
                {
                    "id": 1,
                    "classes_quantity": 2,
                    "task": "Classification",
                    "loss": "CategoricalCrossentropy",
                    "metrics": ["Accuracy"],
                    "callbacks": {
                        "show_every_epoch": True,
                        "plot_loss_metric": True,
                        "plot_metric": True,
                        "plot_loss_for_classes": True,
                        "plot_metric_for_classes": True,
                        "plot_final": True,
                        "show_images": "Best"
                    }
                },
            ],
            "checkpoint": {
                "layer": 1,
                "type": "Metrics",
                "indicator": "Val",
                "mode": "Max",
                "save_best": True,
                "save_weights": False,
            },
        }
    }
}


if __name__ == '__main__':
    traindata = TrainData(**TRAINDATA)

    print(traindata.architecture.outputs_dict[0]["alias"])
    print(traindata.optimizer.parameters.main.learning_rate)

    # print(traindata.json(indent=2))
