"use strict";


(($) => {


    let filters, datasets, params;


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
                    let info = window.TerraProject.datasets[value];
                    if (info) {
                        _dataset = value;
                        params.prepareBtn.disabled = false;
                        params.taskSelect.disabled = false;
                        this.find(".dataset-card-item").removeClass("active");
                        this.find(`.dataset-card[data-name="${_dataset}"]`).parent().addClass("active");
                        params.taskSelect.tasks = info.tasks;
                    } else {
                        _dataset = "";
                        params.prepareBtn.disabled = true;
                        params.taskSelect.disabled = true;
                        this.find(".dataset-card-item").removeClass("active");
                        params.taskSelect.tasks = [];
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

            let _task = "";

            this.prepareBtn = this.find(".actions-form > .prepare > button");
            this.taskSelect = this.find("#dataset-task");

            Object.defineProperty(this, "locked", {
                set: (value) => {
                    let container = $("body.namespace-apps_project main > .container");
                    this.prepareBtn.disabled = value;
                    this.taskSelect.disabled = value;
                    value ? container.addClass("locked") : container.removeClass("locked");
                }
            });

            Object.defineProperty(this, "task", {
                set: (value) => {
                    _task = value;
                    if (_task) {
                        window.StatusBar.message(window.Messages.get("DATASET_SELECTED", [`${datasets.dataset}[${_task}]`]), true);
                    } else {
                        window.StatusBar.message_clear();
                    }
                },
                get: () => {
                    return _task;
                }
            });

            Object.defineProperty(this, "task_name", {
                get: () => {
                    let name = window.TerraProject.tags[_task];
                    return name || "";
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

            Object.defineProperty(this.taskSelect, "disabled", {
                set: (value) => {
                    if (value) this.taskSelect.attr("disabled", "disabled");
                    else this.taskSelect.removeAttr("disabled");
                    $(this.taskSelect).selectmenu("refresh");
                },
                get: () => {
                    return this.taskSelect.attr("disabled") !== undefined;
                }
            });

            Object.defineProperty(this.taskSelect, "tasks", {
                set: (value) => {
                    let options = "";
                    value.forEach((name) => {
                        options = `${options}<option value="${name}">${window.TerraProject.tags[name]}</option>`;
                    });
                    this.task = value.length ? value[0] : "";
                    this.taskSelect.disabled = value.length === 0;
                    $(this.taskSelect).html(options);
                    $(this.taskSelect).selectmenu("refresh");
                }
            });

            this.taskSelect.selectmenu({
                change:(event, ui) => {
                    this.taskSelect.val(ui.item.value).trigger("change");
                }
            }).bind("change", (event) => {
                this.task = event.currentTarget.value;
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
                        is_custom:window.TerraProject.datasets[datasets.dataset].filters.custom !== undefined
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

        }


    });


    $(() => {

        filters = $(".project-datasets-block.filters").DatasetsFilters();
        datasets = $(".project-datasets-block.datasets").DatasetsItems();
        params = $(".properties form.params").DatasetsParams();

        datasets.dataset = window.TerraProject.dataset;

    })


})(jQuery);
