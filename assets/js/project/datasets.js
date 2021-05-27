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
                // this.css("padding-top", `${filters.innerHeight()+1}px`);
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
                        source: "",
                        not_load_layers: false,
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

            let task_type = [
                "classification", "segmentation", "text_segmentation", "regression", "object_detection", "autoencoder", "gan", "timeseries"
            ];

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
                event.preventDefault();
                let serialize_data = this.serializeObject()
                window.ExchangeRequest(
                    "load_dataset",
                    (success, data) => {
                        if (success) {

                            window.StatusBar.message(window.Messages.get("LOAD_DATASET_SUCCESS"), true);

                            $(".inputs-layers").empty();
                            $(".outputs-layers").empty();
                            dataset_params = data.data

                            let params = data.data.audio

                            for(let i=1; i<=serialize_data.num_links.inputs; i++){

                                $(".inputs-layers").append($("<div></div>").addClass("layout-item").addClass("input-layout").attr('name', 'input_' + i).attr('id', 'input_' + i));
                                let input_item = $("#input_"+i)
                                input_item.append($("<div></div>").addClass("layout-title").text("Слой \"input_"+i+"\""));
                                input_item.append($("<div></div>").addClass("layout-params"));

                                let widget = window.FormWidget("inputs[input_" + i + "][name]", {label: "Название входа", type: "str", default: "input_" + i}).addClass("field-inline");
                                input_item.find(".layout-params").append(widget)

                                widget = window.FormWidget("inputs[input_" + i + "][tag]", {label: "Тип данных", type: "str", list: true, available: Object.keys(data.data), default: "audio"}).addClass("field-inline");
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


                            for(let i=1; i<=serialize_data.num_links.outputs; i++){

                                $(".outputs-layers").append($("<div></div>").addClass("layout-item").addClass("output-layout").attr('name', 'output' + i).attr('id', 'output_' + i));
                                let output_item = $("#output_"+i);
                                output_item.append($("<div></div>").addClass("layout-title").text("Слой \"output_"+i+"\""));
                                output_item.append($("<div></div>").addClass("layout-params"));

                                let widget = window.FormWidget("outputs[output_" + i + "][name]", {label: "Название входа", type: "str", default: "output_" + i}).addClass("field-inline");
                                output_item.find(".layout-params").append(widget)

                                widget = window.FormWidget("outputs[output_" + i + "][tag]", {label: "Тип данных", type: "str", list: true, available: Object.keys(data.data), default: "audio"}).addClass("field-inline");
                                output_item.find(".layout-params").append(widget)

                                widget.find("select").selectmenu({
                                    change:(event) => {
                                        $(event.target).trigger("change");
                                    }
                                }).bind("change", (event) => {
                                    output_item.find(".layout-parameters").empty();
                                    let params = dataset_params[$(event.currentTarget).val()];
                                    load_layout_params(output_item, params, "output");
                                })

                                widget = window.FormWidget("outputs[output_" + i + "][task_type]", {label: "Тип задачи", type: "str", list: true, available: task_type}).addClass("field-inline");
                                output_item.find(".layout-params").append(widget)
                                output_item.append($("<div></div>").addClass("layout-parameters"))
                                load_layout_params(output_item, params, "output")
                            }


                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    },
                    serialize_data
                );
            });





            return this;
        },

        DatasetPrepare: function (){

            if (!this.length) return this;

            $( ".slider-range" ).slider({
                range: true,
                min: 0,
                max: 100,
                values: [ 35, 70 ],
                slide: function( event, ui ) {
                    if(ui.values[0] > 90){
                        ui.values[0] = 90;
                        $(".slider-range").slider( "values", 0, 90);
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

            // $("#amount1").val( $( ".slider-range" ).slider( "values", 0));
            // $("#amount2").val( $( ".slider-range" ).slider( "values", 1) - $( ".slider-range" ).slider( "values", 0));
            // $("#amount3").val( 100 - $( ".slider-range" ).slider( "values", 1));

             // $("#amount1").on("input", ()=>{
             //     $(".slider-range").slider( "values", 0, $("#amount1").val())
             // });
             //
             // $("#amount2").on("input", ()=>{
             //     $(".slider-range").slider( "values", 1, $("#amount2").val())
             // });

            $(".number-classes").on("input", (event)=>{
                let num_classes = $(event.target).val();
                let layout = $(event.target).parents('.output-layout')
                if(num_classes < 0 || num_classes > 16){
                    $(event.target).val(16);
                    num_classes = 16;
                }

                layout.find(".class-inline").remove()
                layout.find(".color-inline").remove()

                for(let i=0; i<num_classes; i++){
                    let html = '';
                    html += '<div class="field-form field-inline class-inline">';
                    html += `<label>класс ${i+1}</label>`;
                    html += '<input type="text">';
                    html += '</div>';
                    html += '<div class="field-form field-inline color-inline">';
                    html += '<label>Цвет</label>';
                    html += `<input type="text" class="color-input" value="#123456" />`;
                    html += '<button class="colorpicker-btn"></button>';
                    html += '<div class="colorpicker" hidden></div>';
                    html += `</div>`;
                    layout.find(".layout-params").append(html);
                    layout.find(".colorpicker").last().farbtastic(layout.find(".color-input").last());

                    layout.find(".colorpicker-btn").last().bind("click", (event)=>{
                        event.preventDefault();
                        let field = event.target.parentNode;
                        $(field).find(".colorpicker").last().slideToggle();
                    });
                }
            });

            this.bind("submit", (event)=>{
                event.preventDefault();
                let serialize_data = this.serializeObject();
                console.log(this.serializeObject());
                window.ExchangeRequest(
                    "create_dataset",
                    (success, data) => {
                        if (success) {
                            window.StatusBar.message(window.Messages.get("PRERAPE_DATASET_SUCCESS"), true);

                            console.log(data)

                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    },
                    {
                        dataset_dict: serialize_data
                    }
                );
            });


            return this;
        }


    });


    $(() => {

        filters = $(".project-datasets-block.filters").DatasetsFilters();
        datasets = $(".project-datasets-block.datasets").DatasetsItems();
        params = $(".properties form.params.dataset-change").DatasetsParams();
        dataset_load = $(".dataset-load").DatasetLoad();
        dataset_prepare = $(".dataset-prepare").DatasetPrepare();

        datasets.dataset = window.TerraProject.dataset;

    })


})(jQuery);
