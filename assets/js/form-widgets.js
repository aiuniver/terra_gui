"use strict";


(($) => {


    window.FormWidget = (name, options) => {

        let field_type = "input",
            widget_wrapper = $(`<div class="field-form"></div>`);

        if (options.list === true) field_type = "select";
        if (options.checkbox == true) field_type = "checkbox";

        let render_widget = {

            "input": () => {
                let type;
                switch (options.type) {
                    case "int":
                        type = "number";
                        break;
                    default:
                        type = "text";
                        break;
                }
                return widget_wrapper.append(
                    $(`
                        <label for="field_form-${name}">${name}:</label>
                        <input type="${type}" id="field_form-${name}" name="${name}" value="${options.default || ''}" />
                    `)
                );
            },

            "select": () => {
                let select = $(`<select name="${name}" id="field_form-${name}" class="jquery-ui-menuselect"></select>`);
                for (let index in options.available) {
                    let option = options.available[index] || "";
                    select.append($(`
                        <option value="${option}"${options.default === option ? ' selected="selected"' : ""}>${option}</option>
                    `));
                }
                let widget = widget_wrapper.append(
                    $(`<label for="field_form-${name}">${name}:</label>`), select
                );
                widget.children("select").selectmenu();
                return widget;
            },

            "checkbox": ()=>{
                return widget_wrapper.append(
                    $(`
                        <label for="field_form-${options.name}">${options.name}</label>
                        <div class="checkout-switch">
                            <input type="checkbox" id="field_form-${options.name}" name="${options.name}" value="${options.default}"/>
                            <span class="switcher"></span>
                        </div>
                    `)
                );
            }

        }

        return render_widget[field_type]()

    }


})(jQuery);
