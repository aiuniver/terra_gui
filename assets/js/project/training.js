"use strict";


(($) => {


    let training_toolbar, training_params, training_results;


    let ActionStates = {
        idle: "idle",
        locked: "locked",
        training: "training",
    }


    let RenderOutput = (index, layer) => {
        let layer_block = $(`
            <div class="inner">
                ${layer.name ? `<div class="title">Слой <b>«${layer.name}»</b></div>` : ``}
                <div class="form-inline-label">
                    <input type="hidden" id="field_form-architecture_parameters_outputs_${index}_alias" name="architecture[parameters][outputs][${index}][alias]" value="${layer.alias}" data-value-type="string">
                    <input type="hidden" id="field_form-architecture_parameters_outputs_${index}_task" name="architecture[parameters][outputs][${index}][task]" value="${layer.fields.task.default}" data-value-type="string" data-type="task" data-index="${index}">
                </div>
            </div>
        `);

        Object.defineProperty(layer_block, "task", {
            set: (value) => {
                if (layer_block.find("input[data-type=task]").val() === value) return;
                let metrics = layer_block.find("select[data-type=metrics]");
                layer_block.find("input[data-type=task]").val(value).trigger("input");
                metrics.html("");
                training_params.config.metrics[value].forEach((item) => {
                    metrics.append($(`<option value="${item}">${item}</option>`));
                });
                metrics.selectmenu("refresh");
            }
        });

        layer.fields.loss.available = training_params.config.losses;
        let loss = window.FormWidget(`architecture[parameters][outputs][${index}][loss]`, layer.fields.loss).addClass("field-inline field-reverse");
        loss.find("select").bind("change", (event) => {
            layer_block.task = event.currentTarget[event.currentTarget.selectedIndex].parentNode.label;
        }).selectmenu({
            "change": (event) => {
                $(event.target).trigger("change");
            }
        });
        layer_block.children(".form-inline-label").append(loss);

        let metrics = window.FormWidget(`architecture[parameters][outputs][${index}][metrics][]`, layer.fields.metrics).addClass("field-inline field-reverse");
        metrics.find("select").attr("data-type", "metrics");
        layer_block.children(".form-inline-label").append(metrics);

        let classes_quantity = window.FormWidget(`architecture[parameters][outputs][${index}][classes_quantity]`, layer.fields.classes_quantity).addClass("field-inline field-reverse");
        layer_block.children(".form-inline-label").append(classes_quantity);

        loss.find("select").trigger("change");
        return layer_block;
    }


    let RenderCallback = (index, layer) => {
        return $(`
            <div class="inner">
                ${layer ? `<div class="title">Слой <b>«${layer}»</b></div>` : ``}
                <div class="form-inline-label"></div>
            </div>
        `);
    }


    let RenderCheckpoint = (data) => {
        let block = $(`<div class="inner form-inline-label"></div>`);
        for (let name in data) {
            let widget = window.FormWidget(`architecture[parameters][checkpoint][${name}]`, data[name]).addClass("field-inline field-reverse");
            block.append(widget);
        }
        return block;
    }


    let RenderArchitectureParameters = {
        Basic: (params_block) => {
            let render = {
                outputs: (block, data) => {
                    for (let i in data) {
                        block.append(RenderOutput(i, data[i]));
                    }
                },
                checkpoint: (block, data) => {
                    block.append(RenderCheckpoint(data));
                },
                callbacks: (block, data) => {
                    for (let i in data) {
                        block.append(RenderCallback(i, data[i]));
                    }
                }
            }
            for (let group in training_params.config.architectures.Basic) {
                let cfg = training_params.config.architectures.Basic[group],
                    block = $(`
                        <div class="params-item params-${group}${cfg.collapsable ? ' collapsable' : ''}${cfg.collapsed ? ' collapsed' : ''}">
                            <div class="params-title">${cfg.label}</div>
                        </div>
                    `).CollapsableGroup();
                render[group](block, cfg.data);
                params_block.append(block);
            }
            params_block.find("input[data-type=task]").bind("input", (event) => {
                let block = $(params_block.find(".params-callbacks > .inner")[event.currentTarget.dataset.index]);
                block.children(".form-inline-label").html("");
                let callbacks = training_params.config.callbacks[event.currentTarget.value];
                for (let name in callbacks) {
                    let widget = window.FormWidget(`architecture[parameters][outputs][${event.currentTarget.dataset.index}][callbacks][${name}]`, callbacks[name]).addClass("field-inline field-reverse");
                    block.children(".form-inline-label").append(widget);
                }
            }).trigger("input");
        },
        Yolo: (params_block) => {
            console.log("Render Yolo architecture with parameters", training_params.config);
        }
    }


    $.fn.extend({


        TrainingToolbar: function() {

            if (!this.length) return this;

            return this;

        },


        TrainingParams: function() {

            if (!this.length) return this;

            let _architecture,
                _optimizer,
                _on_training = false,
                _config = JSON.parse($.base64.atob(window._training_form)),
                params_block = this.find(".params-config .mCSB_container");

            Object.defineProperty(this, "config", {
                get: () => {
                    return _config;
                }
            });

            Object.defineProperty(this, "architecture", {
                set: (value) => {
                    _architecture = value;
                    params_block.children(".params-architecture").html("");
                    let method = RenderArchitectureParameters[_architecture];
                    if (method) method(params_block.children(".params-architecture"));
                },
                get: () => {
                    return _architecture;
                }
            });

            Object.defineProperty(this, "optimizer", {
                set: (value) => {
                    _optimizer = value;
                    render_optimizer_parameters(_config.optimizers[_optimizer]);
                },
                get: () => {
                    return _optimizer;
                }
            });

            Object.defineProperty(this, "locked", {
                set: (value) => {
                    if (value === false) this.children(".params-config").removeClass("disabled");
                    else if (value === true) this.children(".params-config").addClass("disabled");
                }
            });

            this.find(".actions-form > *").each((index, item) => {
                Object.defineProperty(item, "disabled", {
                    set: (value) => {
                        if (value === false) $(item).children("button").removeAttr("disabled");
                        else if (value === true) $(item).children("button").attr("disabled", "disabled");
                    }
                });
            });

            Object.defineProperty(this, "ActionState", {
                set: (value) => {
                    let training = this.find(".actions-form > .training")[0],
                        stop = this.find(".actions-form > .stop")[0],
                        reset = this.find(".actions-form > .reset")[0];
                    switch (value) {
                        case ActionStates.idle:
                            training.disabled = false;
                            stop.disabled = true;
                            reset.disabled = true;
                            this.locked = false;
                            break;
                        case ActionStates.locked:
                            training.disabled = true;
                            stop.disabled = true;
                            reset.disabled = true;
                            this.locked = true;
                            break;
                        case ActionStates.training:
                            training.disabled = true;
                            stop.disabled = false;
                            reset.disabled = true;
                            this.locked = true;
                            break;
                    }
                }
            });

            Object.defineProperty(this, "on_training", {
                set: (value) => {
                    if (typeof value !== "boolean") return;
                    _on_training = value;
                },
                get: () => {
                    return _on_training;
                }
            });

            this.bind("submit", (event) => {
                event.preventDefault();
            });

            this.find(".actions-form > .training > button").bind("click", (event) => {
                event.preventDefault();
                if (!this.on_training) {
                    this.on_training = true;
                    this.ActionState = ActionStates.locked;
                    window.ExchangeRequest(
                        "before_start_training",
                        (success, data) => {
                            if (success) {
                                window.TerraProject.logging = data.data.logging;
                                if (data.data.validated) {
                                    this.ActionState = ActionStates.training;
                                    window.ExchangeRequest(
                                        "start_training",
                                        (success, data) => {
                                            if (!success) {
                                                this.on_training = false;
                                                this.ActionState = ActionStates.idle;
                                                window.StatusBar.message(data.error, false);
                                            }
                                        }
                                    );
                                    // window.ExchangeRequest("get_data", this.get_data_response);
                                } else {
                                    $.cookie("model_need_validation", true, {path: window.TerraProject.path.modeling});
                                    window.location = window.TerraProject.path.modeling;
                                }
                            } else {
                                this.on_training = false;
                                this.ActionState = ActionStates.idle;
                                window.StatusBar.message(data.error, false);
                            }
                        },
                        _serialize_training_data()
                    )
                }
            });

            let _serialize_training_data = () => {
                let data = this.serializeObject();

                let _basic_serializer = (data) => {
                    data.architecture.parameters.outputs.forEach((item) => {
                        if (item.callbacks.show_images !== undefined && !item.callbacks.show_images){
                            item.callbacks.show_images = undefined;
                        }
                    });
                    return data;
                }

                switch (this.architecture) {
                    case "Basic":
                        data = _basic_serializer(data);
                        break;
                }

                return data;
            }

            let render_optimizer_parameters = (config) => {
                let template = params_block.children(".params-optimizer-extra");
                template.children(".inner").html("");
                for (let name in config) {
                    let widget = window.FormWidget(`optimizer[parameters][extra][${name}]`, config[name]);
                    widget.addClass("field-inline field-reverse");
                    template.children(".inner").append(widget);
                }
            }

            this.render = () => {
                let template = params_block.find(".params-optimizer"),
                    architecture = window.FormWidget("architecture[type]", this.config.form.architecture),
                    optimizer = window.FormWidget("optimizer[type]", this.config.form.optimizer),
                    batch = window.FormWidget("batch", this.config.form.batch),
                    epochs = window.FormWidget("epochs", this.config.form.epochs),
                    learning_rate = window.FormWidget("optimizer[parameters][main][learning_rate]", this.config.form.learning_rate),
                    fields_inline = $(`<div class="field-form form-inline-label"></div>`);

                batch.addClass("field-inline");
                epochs.addClass("field-inline");
                learning_rate.addClass("field-inline");

                architecture.find("select").bind("change", (event) => {
                    this.architecture = event.currentTarget[event.currentTarget.selectedIndex].dataset.name;
                }).selectmenu({
                    "change": (event) => {
                        $(event.target).trigger("change");
                    }
                });

                optimizer.find("select").bind("change", (event) => {
                    this.optimizer = event.currentTarget[event.currentTarget.selectedIndex].dataset.name;
                }).selectmenu({
                    "change": (event) => {
                        $(event.target).trigger("change");
                    }
                });

                template.children(".inner").append(
                    architecture,
                    optimizer,
                    fields_inline.append(
                        batch,
                        epochs,
                        learning_rate
                    )
                );

                optimizer.find("select").trigger("change");
                architecture.find("select").trigger("change");
                this.ActionState = ActionStates.idle;
            }

            return this;

        },


        TrainingResults: function() {

            if (!this.length) return this;

            return this;

        }


    })


    $(() => {

        if (!window.TerraProject.dataset) {
            let warning = $("#modal-window-warning").ModalWindow({
                title:"Предупреждение!",
                width:300,
                height:174,
                noclose:true,
                callback:(data) => {
                    warning.children(".wrapper").append($(`
                        <p>Для обучения необходимо загрузить датасет.</p>
                        <p><a class="format-link" href="${window.TerraProject.path.datasets}">Загрузить датасет</a></p>
                    `));
                }
            });
            warning.open();
        } else if (!Object.keys(window.TerraProject.layers).length) {
            let warning = $("#modal-window-warning").ModalWindow({
                title:"Предупреждение!",
                width:300,
                height:174,
                noclose:true,
                callback:(data) => {
                    warning.children(".wrapper").append($(`
                        <p>Для обучения необходимо загрузить модель.</p>
                        <p><a class="format-link" href="${window.TerraProject.path.modeling}">Загрузить модель</a></p>
                    `));
                }
            });
            warning.open();
        }

        training_toolbar = $(".project-training-toolbar > .wrapper").TrainingToolbar();
        training_params = $(".project-training-properties > .wrapper > .params > .params-container").TrainingParams();
        training_results = $(".graphics > .wrapper > .tabs-content > .inner > .tabs-item .tab-container").TrainingResults();
        training_params.render();

        window.ExchangeRequest("get_data", training_params.get_data_response);

        $(window).bind("keyup", (event) => {
            if (event.keyCode === 70) {
                console.log($("form.params-container").serializeObject());
            }
        })

    });


})(jQuery);
