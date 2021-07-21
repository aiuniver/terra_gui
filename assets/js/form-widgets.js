"use strict";


(($) => {


    window.FormWidget = (name, options) => {

        let field_type = "input",
            label = options.label ? options.label : name,
            id = name.replace(/[^a-z^0-9^_]+/g, "_"),
            widget_wrapper = $(`<div class="field-form"></div>`);

        if (options.list === true) field_type = "select";

        let render_widget = {

            "input": () => {
                let type, data_value_type, widget;
                switch (options.type) {
                    case "int":
                    case "float":
                        type = "number";
                        data_value_type = "number";
                        break;
                    case "bool":
                        type = "checkbox";
                        data_value_type = "boolean";
                        break;
                    default:
                        type = "text";
                        data_value_type = "string";
                        break;
                }
                if (type === "checkbox") {
                    widget = $(`
                        <label for="field_form-${id}">${label}</label>
                        <div class="checkout-switch">
                            <input type="${type}" id="field_form-${id}" name="${name}"${options.default === true ? 'checked="checked"' : ''}${options.disabled ? ' disabled="disabled"' : ""}${options.readonly ? ' readonly="readonly"' : ""} data-value-type="${data_value_type}" data-unchecked-value="false"/>
                            <span class="switcher"></span>
                        </div>
                    `);
                } else {
                    widget = $(`
                        <label for="field_form-${id}">${label}</label>
                        <input type="${type}" id="field_form-${id}" name="${name}" value="${options.default === undefined || options.default === null ? '' : options.default}"${options.disabled ? ' disabled="disabled"' : ""}${options.readonly ? ' readonly="readonly"' : ""} data-value-type="${data_value_type}"/>
                    `);
                }
                return widget_wrapper.append(widget);
            },

            "select": () => {
                let select = $(`<select name="${name}" id="field_form-${id}" class="jquery-ui-menuselect"${options.disabled ? ' disabled="disabled"' : ""}   data-value-type="string"></select>`);
                let _render_options = (block, values, names) => {
                    for (let index in values) {
                        let option_value = values[index] || "",
                            option_label = option_value;
                        if (values[index] && typeof values[index] === "object") {
                            option_value = values[index][0];
                            option_label = values[index][1];
                        }
                        let option_name = names ? names[index] : undefined;
                        block.append($(`<option${option_name ? ` data-name="${option_name}"` : ""} value="${option_value}"${options.default === option_value ? ' selected="selected"' : ""}>${option_label}</option>`));
                    }
                }
                if (Array.isArray(options.available)) {
                    _render_options(select, options.available, options.available_names);
                } else {
                    for (let name in options.available) {
                        let optgroup = $(`<optgroup label="${name}"></optgroup>`);
                        _render_options(optgroup, options.available[name], options.available_names ? options.available_names[name] : undefined);
                        select.append(optgroup);
                    }
                }
                let widget = widget_wrapper.append(
                    $(`<label for="field_form-${id}">${label}</label>`), select
                );
                widget.children("select").selectmenu();
                return widget;
            }

        }

        return render_widget[field_type]()

    }


})(jQuery);
