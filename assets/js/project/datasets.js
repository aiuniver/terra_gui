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
                        is_custom:window.TerraProject.datasets[datasets.dataset].tags.custom !== undefined
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
                    }
                );
            });

            return this;

        },

        DatasetLoad: function (){

            if (!this.length) return this;


            $(".params-menu ul li").bind("click", (event) => {
                let active_menus = $(".active-menu")
                active_menus.toggleClass("active-menu");

                let closed_inner = $(`.inner#${active_menus.attr("name")}`),
                    active_inner = $(`.inner#${event.target.getAttribute("name")}`);

                active_inner.slideToggle();
                active_inner.removeAttr("disabled");

                $(event.target).toggleClass( "active-menu" );
                closed_inner.slideToggle();
                closed_inner.attr("disabled", "disabled");
            });

            this.bind("submit", (event)=>{
                event.preventDefault();
                console.log(this.serializeObject());
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
                $( "#amount1" ).val(ui.values[0]);
                $( "#amount2" ).val(ui.values[1]);
              }
            });

            $("#amount1").val( $( ".slider-range" ).slider( "values", 0));
            $("#amount2").val( $( ".slider-range" ).slider( "values", 1));

             $("#amount1").on("input", ()=>{
                 $(".slider-range").slider( "values", 0, $("#amount1").val())
             });

             $("#amount2").on("input", ()=>{
                 $(".slider-range").slider( "values", 1, $("#amount2").val())
             });

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
                console.log(this.serializeObject());
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
