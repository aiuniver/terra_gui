"use strict";


(($) => {


    let filters, datasets, params, dataset_load, dataset_prepare;


    $.fn.extend({


        DatasetsFilters: function() {

            if (!this.length) return this;

            let _filters = [];

            Object.defineProperty(this, "filters", {
                set: (value) => {
                    let index = _filters.indexOf(value);
                    if (index === -1) {
                        _filters.push(value);
                        this.find(`li[data-name=${value}]`).addClass("active");
                    } else {
                        delete _filters[index];
                        _filters = _filters.filter(()=>{return true});
                        this.find(`li[data-name=${value}]`).removeClass("active");
                    }
                    datasets.find(".dataset-card-item").removeClass("hidden");
                    if (_filters.length) {
                        datasets.find(".dataset-card-item").addClass("hidden");
                        datasets.find(`.dataset-card-item.${_filters.join(".")}`).removeClass("hidden");
                    }
                },
                get: () => {
                    return _filters;
                }
            });

            this.find("li > span").bind("click", (event) => {
                event.preventDefault();
                this.filters = event.currentTarget.parentNode.dataset.name;
            });

            return this;

        },


        DatasetsItems: function() {

            if (!this.length) return this;

            let _dataset = "";

            let _onWindowResize = () => {
                this.css("padding-top", `${filters.innerHeight()+1}px`);
                this.removeClass("hidden");
            }

            Object.defineProperty(this, "dataset", {
                set: (value) => {
                    if (window.TerraProject.dataset_exists(value)) {
                        _dataset = value;
                        params.prepareBtn.disabled = false;
                        this.find(".dataset-card-item").removeClass("active");
                        this.find(`.dataset-card[data-name="${_dataset}"]`).parent().addClass("active");
                        window.StatusBar.message(window.Messages.get("DATASET_SELECTED", [_dataset]), true);
                    } else {
                        _dataset = "";
                        params.prepareBtn.disabled = true;
                        this.find(".dataset-card-item").removeClass("active");
                        window.StatusBar.message_clear();
                    }
                },
                get: () => {
                    return _dataset;
                }
            });

            this.find(".dataset-card-item").bind("click", (event) => {
                event.preventDefault();
                this.dataset = $(event.currentTarget).children(".dataset-card")[0].dataset.name;
            });

            $(window).bind("resize", _onWindowResize);
            _onWindowResize();

            return this;

        },


        DatasetsParams: function() {

            if (!this.length) return this;

            this.prepareBtn = this.find(".actions-form > .prepare > button");

            Object.defineProperty(this, "locked", {
                set: (value) => {
                    let container = $("body.namespace-apps_project main > .container");
                    this.prepareBtn.disabled = value;
                    value ? container.addClass("locked") : container.removeClass("locked");
                }
            });

            Object.defineProperty(this.prepareBtn, "disabled", {
                set: (value) => {
                    if (value) this.prepareBtn.attr("disabled", "disabled");
                    else this.prepareBtn.removeAttr("disabled");
                },
                get: () => {
                    return this.prepareBtn.attr("disabled") !== undefined;
                }
            });

            this.bind("submit", (event) => {
                event.preventDefault();
                this.locked = true;
                window.StatusBar.clear();
                window.StatusBar.message(window.Messages.get("DATASET_LOADING", [datasets.dataset]));
                window.ExchangeRequest(
                    "prepare_dataset",
                    (success, data) => {
                        this.locked = false;
                        if (success) {
                            window.TerraProject.layers = data.data.layers;
                            window.TerraProject.schema = data.data.schema;
                            window.TerraProject.dataset = data.data.dataset;
                            window.TerraProject.start_layers = data.data.start_layers;
                            window.StatusBar.progress_clear();
                            window.StatusBar.message(window.Messages.get("DATASET_LOADED", [datasets.dataset]), true);
                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    },
                    {
                        dataset:datasets.dataset,
                        is_custom:window.TerraProject.datasets[datasets.dataset].tags.custom_dataset !== undefined
                    }
                );
                window.ExchangeRequest(
                    "get_data",
                    (success, data) => {
                        if (!success) {
                            datasets.dataset = window.TerraProject.dataset;
                            window.StatusBar.message(data.error, false);
                        } else {
                            window.StatusBar.progress(data.data.progress_status.percents, data.data.progress_status.progress_text);
                        }
                    },
                );
            });

            return this;

        },

        DatasetLoad: function (){

            if (!this.length) return this;

            let task_type_input = ['images', 'text', 'audio', 'dataframe'],
                data_type_output = ['images', 'text', 'audio', 'classification', 'segmentation', 'text_segmentation', 'regression', 'timeseries'],
                dataset_params;

            function componentToHex(c) {
                let hex = c.toString(16);
                return hex.length === 1 ? "0" + hex : hex;
            }

            function rgbToHex(r, g, b) {
                return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
            }

            this.loadBtn = this.find(".actions-form > .item.load > button");

            Object.defineProperty(this, "hidden", {
                set: (value) => {
                    if (value) this.addClass("hidden");
                    else this.removeClass("hidden");
                },
                get: () => {
                    return this.hasClass("hidden");
                }
            });

            Object.defineProperty(this.loadBtn, "disabled", {
                set: (value) => {
                    if (value) this.loadBtn.attr("disabled", "disabled");
                    else this.loadBtn.removeAttr("disabled");
                },
                get: () => {
                    return this.loadBtn.attr("disabled") !== undefined;
                }
            });

            Object.defineProperty(this, "locked", {
                set: (value) => {
                    let container = $("body.namespace-apps_project main > .container");
                    this.loadBtn.disabled = value;
                    value ? container.addClass("locked") : container.removeClass("locked");
                }
            });

            this.find("form > .tabs > li").bind("click", (event) => {
                let target = $(event.currentTarget),
                    data_type = target.attr("data-type");
                target.parent().children("li").removeClass("active");
                target.toggleClass("active");
                this.find(".field-mode-type").addClass("hidden");
                $(`.field-mode-${data_type}`).removeClass("hidden");
                $("input[name=mode]").removeAttr("checked");
                $(`input[name=mode][value=${data_type}]`).attr("checked", "checked");
            });

            let _camelize = (text) => {
                let _capitalize = (word) => {
                    return `${word.slice(0, 1).toUpperCase()}${word.slice(1).toLowerCase()}`
                }
                let words = text.split("_"),
                    result = [_capitalize(words[0])];
                words.slice(1).forEach((word) => result.push(word))
                return result.join(" ")
            }

            let load_layout_params = (elem, params, layer)=>{
                for(let name in params){
                    let param = $.extend(true, {}, params[name]);
                    param.label = _camelize(name);
                    param.default = params[name].default;
                    let widget = window.FormWidget(`${layer}s[${elem.attr('id')}][parameters][${name}]`, param);
                    widget.addClass("field-inline field-reverse");
                    if (["embedding", "bag_of_words", "word_to_vec", "word_to_vec_size"].indexOf(name) > -1) {
                        widget.find("input").attr("data-name", name).attr("data-group", `bow_wtv_group-${elem[0].id}`).addClass(`bow_wtv_group-${elem[0].id} bow_wtv_field-${name}`);
                        if (!name.endsWith("_size")) widget.find("input").addClass("bow_wtv_type_checkbox");
                        else widget.find("input").addClass("bow_wtv_type_checkbox");
                    }
                    elem.find(".layout-parameters").append(widget);
                }
                elem.find(".layout-parameters .bow_wtv_type_checkbox").bind("change", (event) => {
                    let item = $(event.currentTarget);
                    if (item[0].checked) {
                        let checkbox = $(`.bow_wtv_type_checkbox.${item.data("group")}`).not(item);
                        for (let i=0; i<checkbox.length; i++) checkbox[i].checked = false;
                        checkbox.trigger("change");
                    }
                    if (item.data("name") === "word_to_vec") {
                        let size = $(`.${item.data("group")}.bow_wtv_field-word_to_vec_size`);
                        if (item[0].checked) size.removeAttr("disabled");
                        else size.attr("disabled", "disabled");
                    }
                });
                elem.find(".layout-parameters .bow_wtv_field-word_to_vec").trigger("change");
            };

            let _dataset_source_loaded = (data, serialize_data) => {
                $(".inputs-layers").empty();
                $(".outputs-layers").empty();
                dataset_params = data;

                let params = data.images

                for(let i=1; i<=serialize_data.num_links.inputs; i++){

                    $(".inputs-layers").append($("<div></div>").addClass("layout-item").addClass("input-layout").attr('name', 'input_' + i).attr('id', 'input_' + i));
                    let input_item = $("#input_"+i)
                    input_item.append($("<div></div>").addClass("layout-title").html(`Слой <b>«input_${i}»</b>`));
                    input_item.append($("<div></div>").addClass("layout-params form-inline-label"));

                    let widget = window.FormWidget("inputs[input_" + i + "][name]", {label: "Название входа", type: "str", default: "input_" + i}).addClass("field-inline field-reverse");
                    input_item.find(".layout-params").append(widget)

                    widget = window.FormWidget("inputs[input_" + i + "][tag]", {label: "Тип данных", type: "str", list: true, available: task_type_input, default: "images"}).addClass("field-inline field-reverse");
                    input_item.find(".layout-params").append(widget)
                    widget.find("select").selectmenu({
                        change:(event) => {
                            $(event.target).trigger("change");
                        }
                    }).bind("change", (event) => {
                        input_item.find(".layout-parameters").empty();
                        let params = dataset_params[$(event.currentTarget).val()];
                        load_layout_params(input_item, params, "input")
                    })

                    input_item.append($("<div></div>").addClass("layout-parameters form-inline-label"));
                    load_layout_params(input_item, params, "input");
                }

                params = data.classification

                for(let i=1; i<=serialize_data.num_links.outputs; i++){

                    $(".outputs-layers").append($("<div></div>").addClass("layout-item").addClass("output-layout").attr('name', 'output' + i).attr('id', 'output_' + i));
                    let output_item = $("#output_"+i);
                    output_item.append($("<div></div>").addClass("layout-title").html(`Слой <b>«output_${i}»</b>`));
                    output_item.append($("<div></div>").addClass("layout-params form-inline-label"));

                    let widget = window.FormWidget("outputs[output_" + i + "][name]", {label: "Название выхода", type: "str", default: "output_" + i}).addClass("field-inline field-reverse");
                    output_item.find(".layout-params").append(widget)

                    widget = window.FormWidget("outputs[output_" + i + "][tag]", {label: "Тип данных", type: "str", list: true, available: data_type_output, default: "classification"}).addClass("field-inline field-reverse");
                    output_item.find(".layout-params").append(widget)

                    widget.find("select").selectmenu({
                        change:(event) => {
                            $(event.target).trigger("change");
                        }
                    }).bind("change", (event) => {
                        let output_item = $("#output_"+i);
                        output_item.find(".layout-parameters").empty();
                        let params = dataset_params[$(event.currentTarget).val()];
                        load_layout_params(output_item, params, "output")
                        if($(event.currentTarget).val() === "segmentation"){
                            $("select[name='outputs[output_" + i + "][parameters][input_type]']").selectmenu({
                                change:(event) => {
                                    $(event.target).trigger("change");
                                }
                            }).bind("change", (event)=>{
                                let segmentation_change = $(event.currentTarget).val();
                                let output_id = $(event.currentTarget).parents(".output-layout").attr("id");
                                let layout = $(event.target).parents('.layout-item');
                                layout.find(".class-inline").remove()
                                layout.find(".color-inline").remove()
                                layout.find(".number-classes").parent().remove()
                                layout.find(".number-classes-auto").parent().remove()
                                layout.find(".selected-file").remove()
                                if(segmentation_change === "Ручной ввод"){
                                    let widget = window.FormWidget("outputs[" + output_id + "][num_classes]", {label: "Количество классов", type: "int"}).addClass("field-inline field-reverse");
                                    widget.find("input").addClass("number-classes");
                                    $("#"+output_id).find(".layout-parameters").append(widget)
                                    $(widget).on("input", (event)=>{
                                        layout.find(".layout-parameters > .class-inline").remove()
                                        layout.find(".layout-parameters > .color-inline").remove()
                                        let num_classes = $(event.target).val();
                                        if(num_classes < 0 || num_classes > 16){
                                            $(event.target).val(16);
                                            num_classes = 16;
                                        }
                                        for(let i=0; i<num_classes; i++){
                                            let html = '';
                                            html += '<div class="field-form field-inline field-reverse class-inline">';
                                            html += `<label for="${output_id}-color-name-${i+1}">Класс ${i+1}</label>`;
                                            html += `<input type="text" id="${output_id}-color-name-${i+1}" class="class_name">`;
                                            html += '</div>';
                                            html += '<div class="field-form field-inline field-reverse color-inline">';
                                            html += `<label for="${output_id}-color-value-${i+1}">Цвет</label>`;
                                            html += `<input type="text" class="color-input" id="${output_id}-color-value-${i+1}" value="#123456"/>`;
                                            html += '<button class="colorpicker-btn"></button>';
                                            html += '<div class="colorpicker" hidden></div>';
                                            html += `</div>`;
                                            layout.find(".layout-parameters").append(html);
                                            layout.find(".colorpicker").last().farbtastic(layout.find(".color-input").last());
                                            layout.find(".colorpicker").last().bind("click", (event)=>{
                                                 $(event.target).parents(".field-inline").find(".colorpicker-btn").css("background-color", $(event.target).parents(".field-inline").find(".color-input").val());
                                            })

                                            layout.find(".colorpicker-btn").last().bind("click", (event)=>{
                                                event.preventDefault();
                                                let field = event.target.parentNode;
                                                $(field).find(".colorpicker").last().slideToggle();
                                            });
                                        }
                                    });

                                }else if(segmentation_change === "Автоматический поиск"){
                                    let html = '';
                                    html += '<div class="field-form field-inline field-reverse class-inline">';
                                    html += `<label for="outputs[${output_id}][num_classes]">Количество классов</label>`;
                                    html += `<input type="text" id="outputs[${output_id}][num_classes]" class="number-classes-auto">`;
                                    html += '</div>';
                                    html += '<div class="field-form field-inline field-reverse class-inline">';
                                    html += `<button class="search-num-classes">Найти</button>`;
                                    html += `</div>`;
                                    layout.find(".layout-parameters").append(html);
                                    layout.find(".search-num-classes").bind("click", (event)=>{
                                        event.preventDefault();
                                        window.ExchangeRequest(
                                            'get_auto_colors',
                                            (success, data) => {
                                                if(success){
                                                    let num = 1;
                                                    layout.find(".class-inline").remove()
                                                    layout.find(".color-inline").remove()
                                                    layout.find(".search-num-classes").parent().remove()
                                                    layout.find(".number-classes-auto").parent().remove()
                                                    for(let i in data.data){
                                                        let html = '',
                                                        rgb = data.data[i]
                                                        html += '<div class="field-form field-inline field-reverse class-inline">';
                                                        html += `<label for="${output_id}-color-name-${num}">класс ${num}</label>`;
                                                        html += `<input type="text" id="${output_id}-color-name-${num}" class="number-classes-auto" value="${i}">`;
                                                        html += '</div>';
                                                        html += '<div class="field-form field-inline field-reverse color-inline">';
                                                        html += `<label for="${output_id}-color-value-${num}">Цвет</label>`;
                                                        html += `<input type="text" id="${output_id}-color-value-${num}" class="color-input" value="${rgbToHex(rgb[0], rgb[1], rgb[2])}" />`;
                                                        html += '<button class="colorpicker-btn"></button>';
                                                        html += '<div class="colorpicker" hidden></div>';
                                                        html += `</div>`;
                                                        layout.find(".layout-parameters").append(html);
                                                        layout.find(".colorpicker").last().farbtastic(layout.find(".color-input").last());

                                                        layout.find(".colorpicker-btn").last().bind("click", (event)=>{
                                                            event.preventDefault();
                                                            let field = event.target.parentNode;
                                                            $(field).find(".colorpicker").last().slideToggle();
                                                        });
                                                        num++;
                                                    }
                                                }else{
                                                    window.StatusBar.message(data.error, false);
                                                }
                                            },
                                            {
                                                name: $("#"+output_id).find("select[name='outputs["+output_id+"][parameters][folder_name]']").val(),
                                                num_classes: parseInt($("#"+output_id).find(".number-classes-auto").val()),
                                                mask_range: parseInt($("#"+output_id).find("input[name='outputs["+output_id+"][parameters][mask_range]']").val()),
                                                txt_file: false
                                            }
                                        )
                                    })
                                }else if(segmentation_change === "Файл аннотации"){
                                    var folder_values = $.map($("#"+output_id).find("select[name='outputs["+output_id+"][parameters][folder_name]'] option") ,function(option) {
                                        return option.value;
                                    });
                                    let widget = window.FormWidget("outputs[output_" + i + "][parameters][selected_file]", {label: "Название файла", type: "str", list: true, available: folder_values, default: ""}).addClass("field-inline field-reverse selected-file");
                                    layout.find(".layout-parameters").append(widget)
                                    let html = '';
                                    html += '<div class="field-form field-inline field-reverse class-inline">';
                                    html += `<button class="search-num-classes">Найти</button>`;
                                    html += `</div>`;
                                    layout.find(".layout-parameters").append(html);
                                    layout.find(".search-num-classes").bind("click", (event)=>{
                                        event.preventDefault();
                                        window.ExchangeRequest(
                                            'get_auto_colors',
                                            (success, data) => {
                                                if(success){
                                                    layout.find(".class-inline").remove()
                                                    layout.find(".color-inline").remove()
                                                    layout.find(".search-num-classes").parent().remove()
                                                    layout.find(".number-classes-auto").parent().remove()
                                                    layout.find(".number-classes").parent().remove()
                                                    for(let i in data.data){
                                                        let html = '',
                                                        rgb = data.data[i]
                                                        html += '<div class="field-form field-inline field-reverse class-inline">';
                                                        html += `<label for="${output_id}-color-name-${i}">класс ${i}</label>`;
                                                        html += `<input type="text" id="${output_id}-color-name-${i}" value="${i}">`;
                                                        html += '</div>';
                                                        html += '<div class="field-form field-inline field-reverse color-inline">';
                                                        html += `<label for="${output_id}-color-value-${i}">Цвет</label>`;
                                                        html += `<input type="text" id="${output_id}-color-value-${i}" class="color-input" value="${rgbToHex(rgb[0], rgb[1], rgb[2])}" />`;
                                                        html += '<button class="colorpicker-btn"></button>';
                                                        html += '<div class="colorpicker" hidden></div>';
                                                        html += `</div>`;
                                                        layout.find(".layout-parameters").append(html);
                                                        layout.find(".colorpicker").last().farbtastic(layout.find(".color-input").last());

                                                        layout.find(".colorpicker-btn").last().bind("click", (event)=>{
                                                            event.preventDefault();
                                                            let field = event.target.parentNode;
                                                            $(field).find(".colorpicker").last().slideToggle();
                                                        });
                                                    }
                                                }else{
                                                    window.StatusBar.message(data.error, false);
                                                }
                                            },
                                            {
                                                name: $("#"+output_id).find("select[name='outputs["+output_id+"][parameters][selected_file]']").val(),
                                                mask_range: parseInt($("#"+output_id).find("input[name='outputs["+output_id+"][parameters][mask_range]']").val()),
                                                txt_file: true
                                            }
                                        )
                                    })
                                }
                            })
                        }
                    })

                    output_item.append($("<div></div>").addClass("layout-parameters form-inline-label"))
                    load_layout_params(output_item, params, "output")
                }

            }

            this.bind("submit", (event)=>{
                this.locked = true;
                event.preventDefault();
                let serialize_data = this.find("form").serializeJSON();
                window.StatusBar.clear();
                window.StatusBar.message(window.Messages.get("DATASET_SOURCE_LOADING"));
                window.ExchangeRequest(
                    "before_load_dataset_source",
                    (success, data) => {
                        if (success) {
                            window.ExchangeRequest(
                                "load_dataset",
                                (success, data) => {
                                    this.locked = false;
                                    if (success) {
                                        window.StatusBar.clear();
                                        window.StatusBar.message(window.Messages.get("DATASET_SOURCE_LOADED"), true);
                                        _dataset_source_loaded(data.data, serialize_data);
                                        dataset_prepare.hidden = false;
                                    } else {
                                        window.StatusBar.message(data.error, false);
                                    }
                                },
                                serialize_data
                            );
                            window.ExchangeRequest(
                                "get_data",
                                (success, data) => {
                                    if (success) {
                                        window.StatusBar.progress(data.data.progress_status.percents, data.data.progress_status.progress_text);
                                    } else {
                                        this.locked = false;
                                        window.StatusBar.message(data.error, false);
                                    }
                                }
                            );
                        } else {
                            this.locked = false;
                            window.StatusBar.message(data.error, false);
                        }
                    }
                );
            });

            return this;
        },

        DatasetPrepare: function (){

            if (!this.length) return this;

            function hexToRgb(hex) {
                var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
                return result ? [
                    parseInt(result[1], 16),
                    parseInt(result[2], 16),
                    parseInt(result[3], 16)
                    ]: null;
            }

            function dataset_tags_string(tags){
                let str = '';
                for(let i in tags){
                    str += ` filter-${i}`
                }
                return str;
            }

            this.createBtn = this.find("button");

            Object.defineProperty(this, "hidden", {
                set: (value) => {
                    if (value) this.addClass("hidden");
                    else this.removeClass("hidden");
                },
                get: () => {
                    return this.hasClass("hidden");
                }
            });

            Object.defineProperty(this.createBtn, "disabled", {
                set: (value) => {
                    if (value) this.createBtn.attr("disabled", "disabled");
                    else this.createBtn.removeAttr("disabled");
                },
                get: () => {
                    return this.createBtn.attr("disabled") !== undefined;
                }
            });

            Object.defineProperty(this, "locked", {
                set: (value) => {
                    let container = $("body.namespace-apps_project main > .container");
                    this.createBtn.disabled = value;
                    value ? container.addClass("locked") : container.removeClass("locked");
                }
            });

            $( ".slider-range" ).slider({
                range: true,
                min: 5,
                max: 98,
                values: [ 60, 90 ],
                slide: function( event, ui ) {
                    if(ui.values[0] > 90){
                        ui.values[0] = 90;
                        $(".slider-range").slider( "values", 0, 90);
                    }else if(ui.values[0] < 5){
                         ui.values[0] = 90;
                        $(".slider-range").slider( "values", 0, 5);
                    }
                    if(ui.values[1] > 95){
                        ui.values[1] = 95;
                        $(".slider-range").slider( "values", 1, 95);
                    };

                    $( "#amount1" ).val(ui.values[0]);
                    $( "#amount2" ).val(ui.values[1] - ui.values[0]);
                    $( "#amount3" ).val(100 - ui.values[1]);
                }
            });

            $("#amount1").val( $( ".slider-range" ).slider( "values", 0));
            $("#amount2").val( $( ".slider-range" ).slider( "values", 1) - $( ".slider-range" ).slider( "values", 0));
            $("#amount3").val( 100 - $( ".slider-range" ).slider( "values", 1));

            this.bind("submit", (event)=>{
                event.preventDefault();
                this.locked = true;
                let layouts = $(".output-layout"),
                    classes_names = {},
                    classes_colors = {};
                for(let i=0; i< layouts.length; i++){
                    let layout_id = layouts[i].id;
                    classes_names[layout_id] = []
                    classes_colors[layout_id] = []
                    let items = $(layouts[i]).find(".color-input")
                    for(let j=0; j<items.length; j++){
                        let item = $(items[j])
                        let key_name = item.parent().prev().find("input").val()
                        classes_names[layout_id].push(key_name)
                        classes_colors[layout_id].push(hexToRgb(item.val()))
                        item.parent().prev().find("input").attr("name", `outputs[${layout_id}][parameters][classes_names]`)
                        item.attr("name", `outputs[${layout_id}][parameters][classes_colors]`)
                    }
                }
                let serialize_data = this.find("form").serializeJSON();

                for(let input in serialize_data.outputs){
                    if(classes_names[input].length != 0 && classes_colors[input].length != 0){
                        serialize_data.outputs[input].parameters.classes_names = classes_names[input]
                        serialize_data.outputs[input].parameters.classes_colors = classes_colors[input]
                    }
                }
                for(let item in serialize_data.outputs){
                    if(!serialize_data.parameters.hasOwnProperty("selected_file")){
                        delete serialize_data.outputs[item].parameters.selected_file;
                    }
                }
                serialize_data.parameters.train_part /= 100
                serialize_data.parameters.val_part /= 100
                serialize_data.parameters.test_part /= 100

                window.StatusBar.clear();
                window.StatusBar.message(window.Messages.get("CREATING_DATASET"));
                window.ExchangeRequest(
                    "before_create_dataset",
                    (success, data) => {
                        if (success) {
                            window.ExchangeRequest(
                                "get_data",
                                (success, data) => {
                                    if (success) {
                                        if(!data.stop_flag){
                                            window.StatusBar.progress(data.data.progress_status.percents, data.data.progress_status.progress_text);
                                        }
                                        if (data.error) {
                                            this.locked = false;
                                            window.StatusBar.message(data.error, false);
                                        }
                                    } else {
                                        this.locked = false;
                                        window.StatusBar.message(data.error, false);
                                    }
                                }
                            );
                            window.ExchangeRequest(
                                "create_dataset",
                                (success, data)=>{
                                    this.locked = false;
                                    if(success){
                                        window.StatusBar.clear();
                                        window.StatusBar.message(window.Messages.get("DATASET_CREATED"), true);
                                        $(".dataset-card-wrapper").empty()
                                        for(let i in data.data.datasets){
                                            let dataset_item = data.data.datasets[i];

                                            let html = '';
                                            html += `<div class="dataset-card-item${ dataset_tags_string(dataset_item.tags) }">`;
                                            html += `<div class="dataset-card" data-name="${ dataset_item.name }">`;
                                            html += `<div class="card-title">${ dataset_item.name }</div>`;
                                            html += '<div class="card-body">';
                                            for(let tag in dataset_item.tags){
                                                html += `<div class="card-tag">${ dataset_item.tags[tag] }</div>`;
                                            }
                                            html += '</div>';
                                            html += ' <div class="card-extra">'
                                            html += ' <div class="wrapper">'
                                            if(dataset_item.size){
                                                html += dataset_item.size
                                            }else{
                                                html += '<span>предустановленный</span>'
                                            }
                                            html += '</div>';
                                            html += '</div>';
                                            html += '</div>';
                                            html += '</div>';
                                            html += '</div>';
                                            $(".dataset-card-wrapper").append(html);
                                        }
                                        window.TerraProject.datasets = data.data.datasets;
                                        $(".project-datasets-block.datasets").DatasetsItems();
                                        datasets.dataset = serialize_data.parameters.name

                                        $(".project-datasets-block.filters").find("ul").empty()
                                        for(let name in data.data.tags){
                                            let tag = data.data.tags[name];
                                            let html = `<li data-name="filter-${ name }"><span>${ tag }</span></li>`;
                                            $(".project-datasets-block.filters").find("ul").append(html);
                                        }
                                        $(".project-datasets-block.filters").find("ul").DatasetsFilters();
                                        $(window).trigger("resize");
                                    } else {
                                        window.StatusBar.message(data.error, false);
                                    }
                                },
                                {
                                    dataset_dict: serialize_data
                                }
                            );
                        } else {
                            this.locked = false;
                            window.StatusBar.message(data.error, false);
                        }
                    }
                );
            });


            return this;
        }


    });


    $(() => {

        params = $(".properties .params-item.dataset-change > form").DatasetsParams();
        filters = $(".project-datasets-block.filters").DatasetsFilters();
        datasets = $(".project-datasets-block.datasets").DatasetsItems();
        dataset_load = $(".load-dataset-field").DatasetLoad();
        dataset_prepare = $(".dataset-prepare").DatasetPrepare();

        datasets.dataset = window.TerraProject.dataset;

    })


})(jQuery);
