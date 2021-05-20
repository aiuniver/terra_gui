"use strict";


(($) => {


    let training_params, training_results;


    $.fn.extend({


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
                    field_num_classes.val(window.TerraProject.training.outputs[output_name].num_classes);
                    field_num_classes.removeAttr("disabled");
                } else {
                    field_num_classes.attr("disabled", "disabled").val(2);
                }
                let inner = $(`.params-callbacks > .callback-${output_name} > .form-inline-label`);
                inner.html("");
                for (let name in callbacks) {
                    let callback = callbacks[name],
                        value = false;
                    try {
                        value = window.TerraProject.training.outputs[output_name].callbacks[name];
                    } catch {}
                    if (!value) value = false;
                    callback.default = value;
                    let widget = window.FormWidget(`outputs[${output_name}][callbacks][${name}]`, callback);
                    widget.addClass("field-inline field-reverse");
                    inner.append(widget);
                }
            }).trigger("change");

            _action_stop.bind("click", (event) => {
                event.preventDefault();
                window.ExchangeRequest(
                    "stop_training",
                    (success, data) => {
                        if (success) {
                            _action_training.removeAttr("disabled");
                            _action_stop.attr("disabled", "disabled");
                            _action_reset.removeAttr("disabled");
                        }
                    }
                )
            });

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
                    window.ExchangeRequest(
                        "before_start_training",
                        (success, output) => {
                            if (success) {
                                if (output.data.validated) {
                                    _action_stop.removeAttr("disabled");
                                    window.ExchangeRequest("start_training", null, data);
                                    window.ExchangeRequest(
                                        "get_data",
                                        (success, data) => {
                                            if (success) {
                                                window.StatusBar.message(data.data.status_string);
                                                window.StatusBar.progress(data.data.progress_status.percents, data.data.progress_status.progress_text);
                                                training_results.charts = data.data.plots;
                                                training_results.texts = data.data.texts;
                                                if (data.stop_flag) {
                                                    this.validate = false;
                                                    _action_training.removeAttr("disabled");
                                                    _action_stop.attr("disabled", "disabled");
                                                    _action_reset.removeAttr("disabled");
                                                }
                                            } else {
                                                this.validate = false;
                                                _action_training.removeAttr("disabled");
                                                _action_stop.attr("disabled", "disabled");
                                                window.StatusBar.message(data.error, false)
                                            }
                                        }
                                    );
                                } else {
                                    $.cookie("model_need_validation", true, {path: window.TerraProject.path.modeling});
                                    window.location = window.TerraProject.path.modeling;
                                }
                            } else {
                                this.validate = false;
                                _action_training.removeAttr("disabled");
                                _action_stop.attr("disabled", "disabled");
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

            Object.defineProperty(this, "charts", {
                get: () => {
                    return this.children(".charts").children(".content");
                },
                set: (charts) => {
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
                }
            });

            Object.defineProperty(this, "texts", {
                get: () => {
                    return this.children(".texts").children(".content");
                },
                set: (texts) => {
                    let map_replace = {
                        '&': '&amp;',
                        '<': '&lt;',
                        '>': '&gt;',
                        '"': '&#34;',
                        "'": '&#39;'
                    };
                    this.texts.children(".inner").html(
                        texts.map((item) => {
                            return `<div class="item"><code>${item.replace(/[&<>'"]/g, (c) => {return map_replace[c]})}</code></div>`;
                        }).join("")
                    );
                }
            });

            Object.defineProperty(this, "scatters", {
                get: () => {
                    return this.children(".scatters").children(".content");
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

        training_params = $(".project-training-properties > .wrapper > .params > .params-container").TrainingParams();
        training_results = $(".graphics > .wrapper > .tabs-content > .inner > .tabs-item .tab-container").TrainingResults();

    });


})(jQuery);
