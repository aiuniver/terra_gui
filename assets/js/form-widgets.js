"use strict";


(($) => {


    window.FormWidget = (name, options) => {

        let field_type = "input",
            label = options.label ? options.label : name,
            widget_wrapper = $(`<div class="field-form"></div>`);

        if (options.list === true) field_type = "select";

        let render_widget = {

            "input": () => {
                let type, widget;
                switch (options.type) {
                    case "int":
                        type = "number";
                        break;
                    case "bool":
                        type = "checkbox";
                        break;
                    default:
                        type = "text";
                        break;
                }
                if (type === "checkbox") {
                    widget = $(`
                        <label for="field_form-${name}">${label}</label>
                        <div class="checkout-switch">
                            <input type="${type}" id="field_form-${name}" name="${name}"${options.default === true ? 'checked="checked"' : ''}${options.disabled ? ' disabled="disabled"' : ""} />
                            <span class="switcher"></span>
                        </div>
                    `);
                } else {
                    widget = $(`
                        <label for="field_form-${name}">${label}:</label>
                        <input type="${type}" id="field_form-${name}" name="${name}" value="${options.default || ''}"${options.disabled ? ' disabled="disabled"' : ""} />
                    `);
                }
                return widget_wrapper.append(widget);
            },

            "select": () => {
                let select = $(`<select name="${name}" id="field_form-${name}" class="jquery-ui-menuselect"${options.disabled ? ' disabled="disabled"' : ""}></select>`);
                for (let index in options.available) {
                    let option = options.available[index] || "";
                    select.append($(`
                        <option value="${option}"${options.default === option ? ' selected="selected"' : ""}>${option}</option>
                    `));
                }
                let widget = widget_wrapper.append(
                    $(`<label for="field_form-${name}">${label}:</label>`), select
                );
                widget.children("select").selectmenu();
                return widget;
            }

        }

        return render_widget[field_type]()

    }


})(jQuery);
