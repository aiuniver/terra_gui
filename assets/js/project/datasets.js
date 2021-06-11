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
                        if (success) {
                            window.TerraProject.layers = data.data.layers;
                            window.TerraProject.schema = data.data.schema;
                            window.TerraProject.dataset = data.data.dataset;
                            window.TerraProject.start_layers = data.data.start_layers;
                            window.StatusBar.progress_clear();
                            window.StatusBar.message(window.Messages.get("DATASET_LOADED", [datasets.dataset]), true);
                            this.locked = false;
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

            let task_type_input = ['images', 'text', 'audio', 'dataframe']

            let task_type_output = [
                'classification', 'segmentation', 'regression', 'timeseries', 'autoencoder'
            ]
            function componentToHex(c) {
                var hex = c.toString(16);
                return hex.length == 1 ? "0" + hex : hex;
            }

            function rgbToHex(r, g, b) {
                return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
            }

            this.loadBtn = this.find("button");

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

            window.ExchangeRequest(
                "get_zipfiles",
                (success, data)=> {
                    if (success) {
                        for(let i in data.data){
                            let option = $(`<option value="${data.data[i]}">${data.data[i]}</option>`);
                            $("#gdrive-select").append(option);
                        }
                         $("#gdrive-select").selectmenu("refresh");
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                }
            );

            $(".load-dataset-field > ul > li").bind("click", (event)=>{
                let target = $(event.target);
                target.parent().children("li").removeClass("active");
                $(".load-dataset-field > .inner > .tab-load-dataset > div").addClass("hidden");
                target.toggleClass("active");
                $("#"+target.attr("data-type")).removeClass("hidden");
            });
            $(".load-dataset-field > ul > li[data-type='div-gdrive']").click();


            let dataset_params;

            let load_layout_params = (elem, params, layer)=>{
                for(let name in params){
                    let param = $.extend(true, {}, params[name]);
                    param.label = name;
                    param.default = params[name].default;
                    let widget = window.FormWidget(`${layer}s[${elem.attr('id')}][parameters][${name}]`, param);
                    widget.addClass("field-inline");
                    elem.find(".layout-parameters").append(widget);
                }
            };

            this.bind("submit", (event)=>{
                this.locked = true;
                event.preventDefault();
                let serialize_data = this.serializeJSON(),
                    mode;
                if(serialize_data.name == ""){
                    mode = "url"
                    serialize_data.name = "name"
                }else{
                    mode = "google_drive"
                }
                window.StatusBar.clear();
                window.StatusBar.message("DATASET_LOADING");
                window.ExchangeRequest(
                    "before_load_dataset_source",
                    (success, data) => {
                        if (success) {
                            window.ExchangeRequest(
                                "get_data",
                                (success, data) => {
                                    if (success) {
                                        window.StatusBar.progress(data.data.progress_status.percents, data.data.progress_status.progress_text);
                                    } else {
                                        window.StatusBar.message(data.error, false);
                                    }
                                }
                            );
                            window.ExchangeRequest(
                                "load_dataset",
                                (success, data) => {
                                    if (success) {
                                        window.StatusBar.clear();
                                        window.StatusBar.message("DATASET_LOADED", true);
            
                                        $(".inputs-layers").empty();
                                        $(".outputs-layers").empty();
                                        dataset_params = data.data
            
                                        let params = data.data.images
            
                                        for(let i=1; i<=serialize_data.num_links.inputs; i++){
            
                                            $(".inputs-layers").append($("<div></div>").addClass("layout-item").addClass("input-layout").attr('name', 'input_' + i).attr('id', 'input_' + i));
                                            let input_item = $("#input_"+i)
                                            input_item.append($("<div></div>").addClass("layout-title").text("Слой \"input_"+i+"\""));
                                            input_item.append($("<div></div>").addClass("layout-params"));
            
                                            let widget = window.FormWidget("inputs[input_" + i + "][name]", {label: "Название входа", type: "str", default: "input_" + i}).addClass("field-inline");
                                            input_item.find(".layout-params").append(widget)
            
                                            widget = window.FormWidget("inputs[input_" + i + "][tag]", {label: "Тип данных", type: "str", list: true, available: task_type_input, default: "images"}).addClass("field-inline");
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
            
                                            input_item.append($("<div></div>").addClass("layout-parameters"));
                                            load_layout_params(input_item, params, "input");
                                        }
            
                                        params = data.data.classification
            
                                        for(let i=1; i<=serialize_data.num_links.outputs; i++){
            
                                            $(".outputs-layers").append($("<div></div>").addClass("layout-item").addClass("output-layout").attr('name', 'output' + i).attr('id', 'output_' + i));
                                            let output_item = $("#output_"+i);
                                            output_item.append($("<div></div>").addClass("layout-title").text("Слой \"output_"+i+"\""));
                                            output_item.append($("<div></div>").addClass("layout-params"));
            
                                            let widget = window.FormWidget("outputs[output_" + i + "][name]", {label: "Название выхода", type: "str", default: "output_" + i}).addClass("field-inline");
                                            output_item.find(".layout-params").append(widget)
            
                                            widget = window.FormWidget("outputs[output_" + i + "][tag]", {label: "Тип данных", type: "str", list: true, available: task_type_output, default: "classification"}).addClass("field-inline");
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
                                                if($(event.currentTarget).val() == "segmentation"){
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
                                                        if(segmentation_change == "Ручной ввод"){
                                                            let widget = window.FormWidget("outputs[" + output_id + "][num_classes]", {label: "Количество классов", type: "int"}).addClass("field-inline");
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
                                                                    html += '<div class="field-form field-inline class-inline">';
                                                                    html += `<label>класс ${i+1}</label>`;
                                                                    html += '<input type="text" class="class_name">';
                                                                    html += '</div>';
                                                                    html += '<div class="field-form field-inline color-inline">';
                                                                    html += '<label>Цвет</label>';
                                                                    html += `<input type="text" class="color-input" value="#123456"/>`;
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
            
                                                        }else if(segmentation_change == "Автоматический поиск"){
                                                            let html = '';
                                                            html += '<div class="field-form field-inline class-inline">';
                                                            html += `<label>Количество классов</label>`;
                                                            html += '<input type="text" class="number-classes-auto">';
                                                            html += '</div>';
                                                            html += '<div class="field-form field-inline class-inline">';
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
                                                                                html += '<div class="field-form field-inline class-inline">';
                                                                                html += `<label>класс ${num}</label>`;
                                                                                html += `<input type="text" class="number-classes-auto" value="${i}">`;
                                                                                html += '</div>';
                                                                                html += '<div class="field-form field-inline color-inline">';
                                                                                html += '<label>Цвет</label>';
                                                                                html += `<input type="text" class="color-input" value="${rgbToHex(rgb[0], rgb[1], rgb[2])}" />`;
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
                                                        }else if(segmentation_change == "Файл аннотации"){
                                                            var folder_values = $.map($("#"+output_id).find("select[name='outputs["+output_id+"][parameters][folder_name]'] option") ,function(option) {
                                                                return option.value;
                                                            });
                                                            let widget = window.FormWidget("outputs[output_" + i + "][parameters][selected_file]", {label: "Название файла", type: "str", list: true, available: folder_values, default: ""}).addClass("field-inline selected-file");
                                                            layout.find(".layout-parameters").append(widget)
                                                            let html = '';
                                                            html += '<div class="field-form field-inline class-inline">';
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
                                                                                html += '<div class="field-form field-inline class-inline">';
                                                                                html += `<label>класс ${i}</label>`;
                                                                                html += `<input type="text" value="${i}">`;
                                                                                html += '</div>';
                                                                                html += '<div class="field-form field-inline color-inline">';
                                                                                html += '<label>Цвет</label>';
                                                                                html += `<input type="text" class="color-input" value="${rgbToHex(rgb[0], rgb[1], rgb[2])}" />`;
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
            
                                            widget = window.FormWidget("outputs[output_" + i + "][task_type]", {label: "Тип задачи", type: "str", list: true, available: task_type_output, default: "classification"}).addClass("field-inline");
                                            output_item.find(".layout-params").append(widget)
                                            output_item.append($("<div></div>").addClass("layout-parameters"))
                                            load_layout_params(output_item, params, "output")
                                        }
            
            
                                    } else {
                                        window.StatusBar.message(data.error, false);
                                    }
                                    this.locked = false;
                                },
                                {
                                    name: serialize_data.name,
                                    mode: mode,
                                    link: serialize_data.link,
                                    num_links: serialize_data.num_links
                                }
            
                            );
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
                let serialize_data = this.serializeJSON();

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
                                    } else {
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
                                            console.log(name, tag)
                                            let html = `<li data-name="filter-${ name }"><span>${ tag }</span></li>`;
                                            $(".project-datasets-block.filters").find("ul").append(html);
                                        }
                                        $(".project-datasets-block.filters").find("ul").DatasetsFilters();
                                    } else{
                                        window.StatusBar.message(data.error, false);
                                    }
                                },
                                {
                                    dataset_dict: serialize_data
                                }
                            );
                        }
                    }
                );
            });


            return this;
        }


    });


      // window.ExchangeRequest(
                                            //     "dataset_created",
                                            //     (success, data) => {
                                            //         window.StatusBar.clear();
                                            //         window.StatusBar.message(window.Messages.get("DATASET_CREATED"), true);
                                            //         // $(".dataset-card-wrapper").empty()
                                            //         // for(let i in data.data.datasets){
                                            //         //     let dataset_item = data.data.datasets[i];
                                            //         //     console.log(dataset_item);
                                            //         //
                                            //         //     let html = '';
                                            //         //     html += `<div class="dataset-card-item${ dataset_tags_string(dataset_item.tags) }">`;
                                            //         //     html += `<div class="dataset-card" data-name="${ dataset_item.name }">`;
                                            //         //     html += `<div class="card-title">${ dataset_item.name }</div>`;
                                            //         //     html += '<div class="card-body">';
                                            //         //     for(let tag in dataset_item.tags){
                                            //         //         html += `<div class="card-tag">${ dataset_item.tags[tag] }</div>`;
                                            //         //     }
                                            //         //     html += '</div>';
                                            //         //     html += '</div>';
                                            //         //     html += '</div>';
                                            //         //     $(".dataset-card-wrapper").append(html);
                                            //         // }
                                            //         // window.TerraProject.datasets = data.data.datasets;
                                            //         // $(".project-datasets-block.datasets").DatasetsItems();
                                            //         // datasets.dataset = serialize_data.parameters.name
                                            //     }
                                            // )


    $(() => {

        filters = $(".project-datasets-block.filters").DatasetsFilters();
        datasets = $(".project-datasets-block.datasets").DatasetsItems();
        params = $(".properties form.params.dataset-change").DatasetsParams();
        dataset_load = $(".dataset-load").DatasetLoad();
        dataset_prepare = $(".dataset-prepare").DatasetPrepare();

        datasets.dataset = window.TerraProject.dataset;



    })


})(jQuery);
