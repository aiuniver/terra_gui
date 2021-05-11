"use strict";


(($) => {


    $.fn.extend({


        TrainingParams: function() {

            if (!this.length) return this;

            let _optimizers = {};

            let _field_optimizer = $("#field_form-optimazer"),
                _field_learning_rate = $("#field_form-learning_rate"),
                _field_output_task = $(".field_form-output_task"),
                _field_output_loss = $(".field_form-output_loss"),
                _field_output_metric = $(".field_form-output_metric"),
                _field_output_num_classes = $(".field_form-output_num_classes"),
                _params_optimazer_extra = $(".params-optimazer-extra");

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
                    _params_optimazer_extra.children(".inner").html("");
                    if (Object.keys(params.extra).length) {
                        let _params = $.extend(true, {}, window.TerraProject.optimizers[value].extra);
                        for (let param in _params) {
                            _params[param].default = _optimizers[value].extra[param];
                            if (!_params[param].label) _params[param].label = _camelize(param);
                            let widget = window.FormWidget(`optimazer[params][extra][${param}]`, _params[param]);
                            widget.addClass("field-inline field-reverse");
                            _params_optimazer_extra.children(".inner").append(widget);
                        }
                        _params_optimazer_extra.removeClass("hidden");
                    } else {
                        _params_optimazer_extra.addClass("hidden");
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
            }).trigger("change");

            this.bind("submit", (event) => {
                event.preventDefault();
                console.log($(event.currentTarget).serializeObject());
            });

            return this;

        }


    })


    $(() => {

        $(".project-training-properties > .wrapper > .params > .params-container").TrainingParams();

    });


})(jQuery);
