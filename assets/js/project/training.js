"use strict";


(($) => {


    $.fn.extend({


        TrainingParams: function() {

            if (!this.length) return this;

            let _optimizers = {};

            let _field_optimizer = $("#field_form-optimizer"),
                _field_learning_rate = $("#field_form-learning_rate"),
                _field_output_loss = $(".field_form-output_loss"),
                _params_optimizer_extra = $(".params-optimizer-extra");

            let _camelize = (text) => {
                let _capitalize = (word) => {
                    return `${word.slice(0, 1).toUpperCase()}${word.slice(1).toLowerCase()}`
                }
                let words = text.split("_"),
                    result = [_capitalize(words[0])];
                words.slice(1).forEach((word) => result.push(word))
                return result.join(" ")
            }

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
                    _field_learning_rate.val(params.main.learning_rate);
                    _params_optimizer_extra.children(".inner").html("");
                    if (Object.keys(params.extra).length) {
                        let _params = $.extend(true, {}, window.TerraProject.optimizers[value].extra);
                        for (let param in _params) {
                            _params[param].default = _optimizers[value].extra[param];
                            if (!_params[param].label) _params[param].label = _camelize(param);
                            let widget = window.FormWidget(`optimizer[params][extra][${param}]`, _params[param]);
                            widget.addClass("field-inline field-reverse");
                            _params_optimizer_extra.children(".inner").append(widget);
                        }
                        _params_optimizer_extra.removeClass("hidden");
                    } else {
                        _params_optimizer_extra.addClass("hidden");
                    }
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
                    field_metric = $(`.field_form-${output_name}-output_metric`),
                    field_num_classes = $(`.field_form-${output_name}-output_num_classes`),
                    metrics = [];
                $(`.field_form-${output_name}-output_task`).val(task);
                field_metric.html("");
                try {
                    metrics = window.TerraProject.compile[task].metrics;
                } catch {}
                if (metrics) {
                    metrics.forEach((item) => {
                        let option = $(`<option value="${item}">${item}</option>`),
                            metric = [];
                        try {
                            metric = window.TerraProject.training.outputs[output_name].metric;
                        } catch {}
                        if (metric.indexOf(item) > -1) option.attr("selected", "selected");
                        field_metric.append(option);
                    });
                    field_metric.removeAttr("disabled");
                } else {
                    field_metric.attr("disabled", "disabled");
                }
                field_metric.selectmenu("refresh");
                if (["classification", "segmentation"].indexOf(task) > -1) {
                    field_num_classes.val(window.TerraProject.training.outputs[output_name].num_classes);
                    field_num_classes.removeAttr("disabled");
                } else {
                    field_num_classes.attr("disabled", "disabled").val(2);
                }
            }).trigger("change");

            this.bind("submit", (event) => {
                event.preventDefault();
                window.StatusBar.clear();
                // let form = $(event.currentTarget).serializeObject(),
                //     data = {};
                // let process_data = (o, data) => {
                //     for (let name in o) {
                //         console.log(name, o[name]);
                //     }
                // }
                // data = process_data()
                // console.log(data);
                // console.log(window.TerraProject.training);
                window.ExchangeRequest(
                    "start_training",
                    (success, data) => {
                        if (success) {
                            // console.log("SUCCESS:", success);
                            // console.log("DATA:", data);
                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    },
                    $(event.currentTarget).serializeObject()
                )
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
