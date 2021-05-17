"use strict";


(($) => {


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
                _action_evaluate = $(".params-container .actions-form > .evaluate > button");

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
                    if (value) {
                        _action_training.attr("disabled", "disabled");
                    } else {
                        _action_training.removeAttr("disabled");
                    }
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

            this.bind("submit", (event) => {
                event.preventDefault();
                if (!this.validate) {
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
                        (success, data) => {
                            this.validate = false;
                            if (success) {
                                if (data.data.validation_errors) {
                                    $.cookie("model_need_validation", true, {path: window.TerraProject.path.modeling});
                                    window.location = window.TerraProject.path.modeling;
                                } else {
                                    window.ExchangeRequest("start_training", null, data);
                                    window.ExchangeRequest(
                                        "get_data",
                                        (success, data) => {
                                            console.log("SUCCESS:", success, ", DATA:", data);
                                            console.log("STOP_FLAG:", data.stop_flag);
                                            console.log("DATA:", data.data);
                                            console.log("===============================");
                                            // if (!success) {
                                            //     datasets.dataset = window.TerraProject.dataset;
                                            //     window.StatusBar.message(data.error, false);
                                            // } else {
                                            //     window.StatusBar.progress(data.data.progress_status.percents, data.data.progress_status.progress_text);
                                            // }
                                        }
                                    );
                                }
                            } else {
                                window.StatusBar.message(data.error, false);
                            }
                        },
                        data
                    )
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

        $(".project-training-properties > .wrapper > .params > .params-container").TrainingParams();

    });


})(jQuery);
