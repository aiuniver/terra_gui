"use strict";


(($) => {


    let training_toolbar, training_params, training_results;


    $.fn.extend({


        TrainingToolbar: function() {

            if (!this.length) return this;

            Object.defineProperty(this, "items", {
                get: () => {
                    return this.find(".menu-section > li");
                }
            });

            Object.defineProperty(this, "btn", {
                get: () => {
                    return {
                        "charts":this.find(".menu-section > li[data-type=charts]")[0],
                        "images":this.find(".menu-section > li[data-type=images]")[0],
                        "texts":this.find(".menu-section > li[data-type=texts]")[0],
                        "scatters":this.find(".menu-section > li[data-type=scatters]")[0],
                    };
                }
            });

            this.items.children("span").bind("click", (event) => {
                event.preventDefault();
                let item = $(event.currentTarget).parent()[0];
                if (!item.disabled) item.active = !item.active;
            });

            this.items.each((index, item) => {
                Object.defineProperty(item, "disabled", {
                    set: (value) => {
                        if (value) {
                            $(item).attr("disabled", "disabled");
                            item.active = false;
                        } else {
                            $(item).removeAttr("disabled");
                        }
                    },
                    get: () => {
                        return item.hasAttribute("disabled");
                    }
                });
                Object.defineProperty(item, "active", {
                    set: (value) => {
                        if (value) {
                            $(item).addClass("active");
                            training_results.children(`.${$(item).data("type")}`).removeClass("hidden");
                        } else {
                            $(item).removeClass("active");
                            training_results.children(`.${$(item).data("type")}`).addClass("hidden");
                        }
                        $(item).find("img").attr("src", `/assets/imgs/training-${$(item).data("type")}${value ? "-active" : ""}.svg`);
                    },
                    get: () => {
                        return $(item).hasClass("active");
                    }
                });
            });

            return this;

        },


        TrainingParams: function() {

            if (!this.length) return this;

            let _validate = false,
                _optimizers = {};

            let _field_optimizer = $("#field_form-optimizer"),
                _field_learning_rate = $("#field_form-learning_rate"),
                _field_output_loss = $(".field_form-output_loss"),
                _params_optimizer_extra = $(".params-optimizer-extra"),
                _action_training = $(".params-container .actions-form > .training > button"),
                _action_stop = $(".params-container .actions-form > .stop > button"),
                _action_reset = $(".params-container .actions-form > .reset > button");

            let _camelize = (text) => {
                let _capitalize = (word) => {
                    return `${word.slice(0, 1).toUpperCase()}${word.slice(1).toLowerCase()}`
                }
                let words = text.split("_"),
                    result = [_capitalize(words[0])];
                words.slice(1).forEach((word) => result.push(word))
                return result.join(" ")
            }

            Object.defineProperty(this, "validate", {
                set: (value) => {
                    _validate = value;
                },
                get: () => {
                    return _validate;
                }
            });

            Object.defineProperty(this, "optimizer", {
                set: (value) => {
                    let _get_defaults = (params) => {
                        let output = {}
                        for (let param in params) {
                            output[param] = {};
                            for (let name in params[param]) {
                                output[param][name] = params[param][name].default;
                            }
                        }
                        return output;
                    }
                    let params = _optimizers[value] ? _optimizers[value] : _get_defaults(window.TerraProject.optimizers[value]);
                    _optimizers[value] = params;
                    _field_learning_rate.val(
                        window.TerraProject.training.optimizer.name === value && window.TerraProject.training.optimizer.params.main.learning_rate !== undefined
                            ? window.TerraProject.training.optimizer.params.main.learning_rate
                            : params.main.learning_rate
                    );
                    _params_optimizer_extra.children(".inner").html("");
                    if (Object.keys(params.extra).length) {
                        let _params = $.extend(true, {}, window.TerraProject.optimizers[value].extra);
                        for (let param in _params) {
                            _params[param].default = window.TerraProject.training.optimizer.name === value && window.TerraProject.training.optimizer.params.extra[param] !== undefined
                                ? window.TerraProject.training.optimizer.params.extra[param]
                                : _optimizers[value].extra[param];
                            if (!_params[param].label) _params[param].label = _camelize(param);
                            let widget = window.FormWidget(`optimizer[params][extra][${param}]`, _params[param]);
                            widget.addClass("field-inline field-reverse");
                            _params_optimizer_extra.children(".inner").append(widget);
                        }
                        _params_optimizer_extra.removeClass("hidden");
                    } else {
                        _params_optimizer_extra.addClass("hidden");
                    }
                    window.TerraProject.training.optimizer.name = value;
                    window.TerraProject.training.optimizer.params = _optimizers[value];
                }
            });

            _field_optimizer.selectmenu({
                change:(event) => {
                    $(event.target).trigger("change");
                }
            }).bind("change", (event) => {
                this.optimizer = $(event.currentTarget).val();
            }).trigger("change");

            _field_output_loss.selectmenu({
                change:(event) => {
                    $(event.target).trigger("change");
                }
            }).bind("change", (event) => {
                let item = $(event.currentTarget),
                    output_name = item.data("output"),
                    task = event.currentTarget.selectedOptions[0].parentNode.label,
                    field_metrics = $(`.field_form-${output_name}-output_metrics`),
                    field_num_classes = $(`.field_form-${output_name}-output_num_classes`),
                    callbacks = window.TerraProject.callbacks[task] === undefined ? {} : window.TerraProject.callbacks[task],
                    metrics = [];
                $(`.field_form-${output_name}-output_task`).val(task);
                field_metrics.html("");
                try {
                    metrics = window.TerraProject.compile[task].metrics;
                } catch {}
                if (metrics) {
                    metrics.forEach((item) => {
                        let option = $(`<option value="${item}">${item}</option>`),
                            metrics = [];
                        try {
                            metrics = window.TerraProject.training.outputs[output_name].metrics;
                        } catch {}
                        if (metrics.indexOf(item) > -1) option.attr("selected", "selected");
                        field_metrics.append(option);
                    });
                    field_metrics.removeAttr("disabled");
                } else {
                    field_metrics.attr("disabled", "disabled");
                }
                field_metrics.selectmenu("refresh");
                if (["classification", "segmentation"].indexOf(task) > -1) {
                    field_num_classes.closest(".field-form").removeClass("hidden");
                } else {
                    field_num_classes.closest(".field-form").addClass("hidden");
                }
                let inner = $(`.params-callbacks > .callback-${output_name} > .form-inline-label`);
                inner.html("");
                for (let name in callbacks) {
                    let callback = callbacks[name],
                        value = false;
                    try {
                        value = window.TerraProject.training.outputs[output_name].callbacks[name];
                    } catch {
                        value = callback.default;
                    }
                    let widget = window.FormWidget(`outputs[${output_name}][callbacks][${name}]`, callback);
                    widget.addClass("field-inline field-reverse");
                    widget.find("input").addClass(`_callback_${name}`).bind("change", (event) => {
                        let input = $(event.currentTarget);
                        if (input.hasClass("_callback_show_best_images") || input.hasClass("_callback_show_worst_images")) {
                            let best = this.find("._callback_show_best_images"),
                                worst = this.find("._callback_show_worst_images");
                            if (input.hasClass("_callback_show_best_images")) {
                                if (best[0].checked) worst[0].checked = false;
                            } else {
                                if (worst[0].checked) best[0].checked = false;
                            }
                        }
                    });
                    inner.append(widget);
                }
            }).trigger("change");

            _action_stop.bind("click", (event) => {
                event.preventDefault();
                _action_stop.attr("disabled", "disabled");
                window.ExchangeRequest(
                    "stop_training",
                    (success, data) => {
                        if (success) {
                            this.validate = false;
                            _action_training.attr("disabled", "disabled");
                            _action_stop.attr("disabled", "disabled");
                            _action_reset.attr("disabled", "disabled");
                        }
                    }
                )
            });

            _action_reset.bind("click", (event) => {
                event.preventDefault();
                window.StatusBar.clear();
                window.ExchangeRequest(
                    "reset_training",
                    (success, data) => {
                        if (success) {
                            this.validate = false;
                            training_results.charts = [];
                            training_results.images = [];
                            training_results.texts = {};
                            training_results.scatters = [];
                            _action_training.text("Обучить");
                            window.StatusBar.message(window.Messages.get("TRAINING_DISCARDED"), true);
                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    }
                );
            });

            this.get_data_response = (success, data) => {
                _action_training.text(data.data.in_training ? "Возобновить" : "Обучить");
                if (success) {
                    if (data.data.errors) {
                        this.validate = false;
                        _action_training.removeAttr("disabled");
                        _action_stop.attr("disabled", "disabled");
                        _action_reset.removeAttr("disabled");
                        window.StatusBar.message(data.data.errors, false);
                        training_params.children(".params-config").removeClass("disabled");
                    } else {
                        _action_training.attr("disabled", "disabled");
                        if (data.data.user_stop_train) _action_stop.attr("disabled", "disabled");
                        else _action_stop.removeAttr("disabled");
                        _action_reset.attr("disabled", "disabled");
                        window.StatusBar.message(data.data.status_string);
                        window.StatusBar.progress(data.data.progress_status.percents, data.data.progress_status.progress_text);
                        training_results.charts = data.data.plots;
                        training_results.images = data.data.images;
                        training_results.texts = data.data.texts;
                        training_results.scatters = data.data.scatters;
                        if (data.stop_flag) {
                            this.validate = false;
                            _action_training.removeAttr("disabled");
                            _action_stop.attr("disabled", "disabled");
                            _action_reset.removeAttr("disabled");
                            training_params.children(".params-config").removeClass("disabled");
                        }
                    }
                } else {
                    this.validate = false;
                    _action_training.removeAttr("disabled");
                    _action_stop.attr("disabled", "disabled");
                    _action_reset.removeAttr("disabled");
                    window.StatusBar.message(data.error, false);
                    training_params.children(".params-config").removeClass("disabled");
                }
            }

            this.bind("submit", (event) => {
                event.preventDefault();
                if (!this.validate) {
                    _action_training.attr("disabled", "disabled");
                    _action_stop.removeAttr("disabled");
                    _action_reset.attr("disabled", "disabled");
                    this.validate = true;
                    window.StatusBar.clear();
                    let data = $(event.currentTarget).serializeObject();
                    for (let param_name in window.TerraProject.optimizers[data.optimizer.name].extra) {
                        let param = window.TerraProject.optimizers[data.optimizer.name].extra[param_name];
                        if (!data.optimizer.params.extra) data.optimizer.params.extra = {};
                        switch (param.type) {
                            case "bool":
                                data.optimizer.params.extra[param_name] = data.optimizer.params.extra[param_name] !== undefined;
                                break;
                        }
                    }
                    for (let group in window.TerraProject.optimizers[data.optimizer.name]) {
                        for (let param in window.TerraProject.optimizers[data.optimizer.name][group]) {
                            switch (window.TerraProject.optimizers[data.optimizer.name][group][param].type) {
                                case "int":
                                    data.optimizer.params[group][param] = parseInt(data.optimizer.params[group][param]);
                                    break;
                                case "float":
                                    data.optimizer.params[group][param] = parseFloat(data.optimizer.params[group][param]);
                                    break;
                            }
                        }
                    }
                    for (let output_name in data.outputs) {
                        data.outputs[output_name].num_classes = $(`.field_form-${output_name}-output_num_classes`).val();
                        let task = data.outputs[output_name].task,
                            callbacks = data.outputs[output_name].callbacks;
                        if (!callbacks) callbacks = {}
                        for (let name in window.TerraProject.callbacks[task]) {
                            let value = callbacks[name];
                            switch (window.TerraProject.callbacks[task][name].type) {
                                case "bool":
                                    callbacks[name] = value !== undefined;
                                    break;
                            }
                        }
                        data.outputs[output_name].callbacks = callbacks;
                    }
                    data.checkpoint.save_best = data.checkpoint.save_best !== undefined;
                    data.checkpoint.save_weights = data.checkpoint.save_weights !== undefined;
                    window.StatusBar.message(window.Messages.get("VALIDATE_MODEL"));
                    training_params.children(".params-config").addClass("disabled");
                    window.ExchangeRequest(
                        "before_start_training",
                        (success, output) => {
                            if (success) {
                                window.TerraProject.logging = output.data.logging;
                                if (output.data.validated) {
                                    _action_stop.removeAttr("disabled");
                                    window.ExchangeRequest("start_training");
                                    window.ExchangeRequest("get_data", this.get_data_response);
                                } else {
                                    $.cookie("model_need_validation", true, {path: window.TerraProject.path.modeling});
                                    window.location = window.TerraProject.path.modeling;
                                }
                            } else {
                                this.validate = false;
                                _action_training.removeAttr("disabled");
                                _action_stop.attr("disabled", "disabled");
                                training_params.children(".params-config").removeClass("disabled");
                                window.StatusBar.message(output.error, false);
                            }
                        },
                        data
                    )
                }
            });

            return this;

        },


        TrainingResults: function() {

            if (!this.length) return this;

            let _camelize = (text) => {
                let _capitalize = (word) => {
                    return `${word.slice(0, 1).toUpperCase()}${word.slice(1).toLowerCase()}`
                }
                let words = text.split("_"),
                    result = [_capitalize(words[0])];
                words.slice(1).forEach((word) => result.push(word))
                return result.join(" ")
            }

            Object.defineProperty(this, "charts", {
                get: () => {
                    return this.children(".charts").children(".content");
                },
                set: (charts) => {
                    if (charts.length) {
                        let disabled = training_toolbar.btn.charts.disabled;
                        training_toolbar.btn.charts.disabled = false;
                        if (disabled) training_toolbar.btn.charts.active = true;
                    } else {
                        training_toolbar.btn.charts.disabled = true;
                    }
                    this.charts.children(".inner").html(charts.length ? '<div class="wrapper"></div>' : '');
                    charts.forEach((item) => {
                        let div = $('<div class="item"><div></div></div>');
                        this.charts.children(".inner").children(".wrapper").append(div);
                        Plotly.newPlot(
                            div.children("div")[0],
                            item.list,
                            {
                                autosize:true,
                                margin:{
                                    l:70,
                                    r:20,
                                    t:60,
                                    b:20,
                                    pad:0,
                                    autoexpand:true,
                                },
                                font:{
                                    color:"#A7BED3"
                                },
                                showlegend:true,
                                legend:{
                                    orientation:"h",
                                    font:{
                                        family:"Open Sans",
                                        color:"#A7BED3",
                                    }
                                },
                                paper_bgcolor:"transparent",
                                plot_bgcolor:"transparent",
                                title:item.title,
                                xaxis:{
                                    showgrid:true,
                                    zeroline:false,
                                    linecolor:"#A7BED3",
                                    gridcolor:"#0E1621",
                                    gridwidth:1,
                                },
                                yaxis:{
                                    title:item.yaxis.title,
                                    showgrid:true,
                                    zeroline:false,
                                    linecolor:"#A7BED3",
                                    gridcolor:"#0E1621",
                                    gridwidth:1,
                                },
                            },
                            {
                                responsive:true,
                                displayModeBar:false,
                            }
                        );
                    });
                }
            });

            Object.defineProperty(this, "images", {
                get: () => {
                    return this.children(".images").children(".content");
                },
                set: (images) => {
                    if (Object.keys(images).length) {
                        let disabled = training_toolbar.btn.images.disabled;
                        training_toolbar.btn.images.disabled = false;
                        if (disabled) training_toolbar.btn.images.active = true;
                    } else {
                        training_toolbar.btn.images.disabled = true;
                    }
                    console.log(images);
                    this.images.html("");
                    for (let name in images) {
                        let group = images[name],
                            group_block = $(`<div class="group"><div class="title">${_camelize(group.title)}</div><div class="inner"></div></div>`);
                        if (group.values && group.values.length) {
                            group.values.forEach((item) => {
                                let item_block = $(`<div class="item"><div class="wrapper"><img src="data:image/png;base64,${item.image}" alt="" /></div></div>`);
                                if (item.title) {
                                    item_block.children(".wrapper").append($(`<div class="title">${item.title}</div>`));
                                }
                                if (item.info && item.info.length) {
                                    let info_block = $('<div class="info"></div>');
                                    item.info.forEach((info) => {
                                        if (info.value) {
                                            info_block.append($(`<div class="param"><label>${info.label}: </label><span>${info.value}</span></div>`));
                                        }
                                    });
                                    item_block.children(".wrapper").append(info_block);
                                }
                                group_block.children(".inner").append(item_block);
                            });
                            this.images.append(group_block);
                        }
                    }
                }
            });

            Object.defineProperty(this, "texts", {
                get: () => {
                    return this.children(".texts").children(".content");
                },
                set: (texts) => {
                    if (Object.keys(texts).length && (texts.summary || texts.epochs.length)) {
                        let disabled = training_toolbar.btn.texts.disabled;
                        training_toolbar.btn.texts.disabled = false;
                        if (disabled) training_toolbar.btn.texts.active = true;
                    } else {
                        training_toolbar.btn.texts.disabled = true;
                    }
                    this.texts.html('<div class="inner"></div>');
                    let format_epoch_value = (value) => {
                        value = `${Math.round(value*1000)/1000}`;
                        let split_value = value.split(".");
                        if (split_value.length === 2) {
                            split_value[0] = `<span>${split_value[0]}</span>`;
                            value = split_value.join("<i>.</i>");
                        }
                        return value;
                    }
                    if (Object.keys(texts).length) {
                        if (texts.epochs.length) {
                            let epochs_block = $('<div class="epochs"><table><thead><tr class="outputs_heads"><th rowspan="2">Эпоха</th><th rowspan="2">Время (сек.)</th></tr><tr class="callbacks_heads"></tr></thead><tbody></tbody></div>'),
                                outputs_cols = {},
                                outputs_list = [];
                            this.texts.children(".inner").append(epochs_block);
                            texts.epochs.forEach((epoch) => {
                                for (let output_name in epoch.data) {
                                    if (!outputs_cols[output_name]) outputs_cols[output_name] = [];
                                    outputs_cols[output_name] = $.merge(outputs_cols[output_name], Object.keys(epoch.data[output_name])).filter((item, i, items) => {
                                        return i === items.indexOf(item);
                                    });
                                }
                            });
                            for (let output_name in outputs_cols) {
                                let callbacks_cols = outputs_cols[output_name];
                                outputs_list.push(output_name);
                                epochs_block.find(".outputs_heads").append($(`<th colspan="${callbacks_cols.length}">${output_name}</th>`));
                                callbacks_cols.forEach((callback_name) => {
                                    epochs_block.find(".callbacks_heads").append($(`<th>${callback_name}</th>`));
                                });
                            }
                            texts.epochs.forEach((epoch) => {
                                let tr = $(`<tr><td class="epoch_num">${epoch.number}</td><td>${epoch.time}</td></tr>`);
                                outputs_list.forEach((output_name) => {
                                    outputs_cols[output_name].forEach((callback_name) => {
                                        let td = $(`<td class="value"><code></code></td>`);
                                        try {
                                            td.html(format_epoch_value(epoch.data[output_name][callback_name]));
                                        } catch (e) {}
                                        tr.append(td);
                                    });
                                });
                                epochs_block.find("tbody").append(tr);
                            });
                        }
                        if (texts.summary) {
                            this.texts.children(".inner").append(`<div class="summary">${texts.summary}</div>`);
                        }
                    }
                }
            });

            Object.defineProperty(this, "scatters", {
                get: () => {
                    return this.children(".scatters").children(".content");
                },
                set: (scatters) => {
                    if (scatters.length) {
                        let disabled = training_toolbar.btn.scatters.disabled;
                        training_toolbar.btn.scatters.disabled = false;
                        if (disabled) training_toolbar.btn.scatters.active = true;
                    } else {
                        training_toolbar.btn.scatters.disabled = true;
                    }
                    this.scatters.children(".inner").html(scatters.length ? '<div class="wrapper"></div>' : '');
                    scatters.forEach((item) => {
                        let div = $('<div class="item"><div></div></div>');
                        this.scatters.children(".inner").children(".wrapper").append(div);
                        Plotly.newPlot(
                            div.children("div")[0],
                            item.list,
                            {
                                autosize:true,
                                margin:{
                                    l:70,
                                    r:20,
                                    t:60,
                                    b:20,
                                    pad:0,
                                    autoexpand:true,
                                },
                                font:{
                                    color:"#A7BED3"
                                },
                                showlegend:true,
                                legend:{
                                    orientation:"h",
                                    font:{
                                        family:"Open Sans",
                                        color:"#A7BED3",
                                    }
                                },
                                paper_bgcolor:"transparent",
                                plot_bgcolor:"transparent",
                                title:item.title,
                                xaxis:{
                                    showgrid:true,
                                    zeroline:false,
                                    linecolor:"#A7BED3",
                                    gridcolor:"#0E1621",
                                    gridwidth:1,
                                },
                                yaxis:{
                                    title:item.yaxis.title,
                                    showgrid:true,
                                    zeroline:false,
                                    linecolor:"#A7BED3",
                                    gridcolor:"#0E1621",
                                    gridwidth:1,
                                },
                            },
                            {
                                responsive:true,
                                displayModeBar:false,
                            }
                        );
                    });
                }
            });

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

        window.ExchangeRequest("get_data", training_params.get_data_response);

    });


})(jQuery);
