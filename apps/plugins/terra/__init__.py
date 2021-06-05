from .exchange import TerraExchange

terra_exchange = TerraExchange()

if terra_exchange.project.dataset:
    dataset_name = terra_exchange.project.dataset
    is_custom = False
    for dataset in terra_exchange.project.datasets:
        if dataset.name == dataset_name:
            is_custom = "custom_dataset" in dataset.tags.keys()
    layers = terra_exchange.project.dict().get("layers")
    terra_exchange.call(
        "prepare_dataset",
        dataset=terra_exchange.project.dataset,
        is_custom=is_custom,
        not_load_layers=False
    )
    terra_exchange.call("autosave_project")
