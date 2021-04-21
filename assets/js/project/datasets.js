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
                        this.find(`.dataset-card[data-name=${_dataset}]`).parent().addClass("active");
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
                window.StatusBar.message(window.Messages.get("DATASET_LOADING", [`${datasets.dataset} [${this.task_name}]`]));
                window.ExchangeRequest(
                    "prepare_dataset",
                    (success, data) => {
                        if (success) {
                            window.TerraProject.dataset = datasets.dataset;
                            window.TerraProject.task = this.task;
                            window.StatusBar.progress_clear();
                            window.StatusBar.message(window.Messages.get("DATASET_LOADED", [`${datasets.dataset} [${this.task_name}]`]), true);
                            this.locked = false;
                        }
                    },
                    {
                        dataset:datasets.dataset,
                        task:this.task_name,
                        is_custom:window.TerraProject.datasets[datasets.dataset].filters.custom !== undefined
                    }
                );
                window.ExchangeRequest(
                    "get_data",
                    (success, data) => {
                        if (!success) {
                            datasets.dataset = window.TerraProject.dataset;
                            this.task = window.TerraProject.task;
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


    // let _isLoaded = true;
    //
    //
    // let onResize = () => {
    //     let datasets = $(".project-datasets-block.datasets");
    //     datasets.css("padding-top", `${$(".project-datasets-block.filters").innerHeight()+1}px`);
    //     datasets.removeClass("hidden");
    // }
    //
    //
    // let LockedForm = (locked) => {
    //     let button = $(".params-container .params-item > .inner > .actions-form > .prepare > button"),
    //         select = $("#dataset-task"),
    //         datasets = $(".dataset-card-item").not(".active").children(".dataset-card");
    //     if (locked) {
    //         button.attr("disabled", "disabled");
    //         select.attr("disabled", "disabled").selectmenu("refresh");
    //         datasets.addClass("disabled");
    //     } else {
    //         button.removeAttr("disabled");
    //         select.removeAttr("disabled").selectmenu("refresh");
    //         datasets.removeClass("disabled");
    //     }
    // }


    $(() => {

        filters = $(".project-datasets-block.filters").DatasetsFilters();
        datasets = $(".project-datasets-block.datasets").DatasetsItems();
        params = $(".properties form.params").DatasetsParams();

        datasets.dataset = window.TerraProject.dataset;

        // $(".project-datasets-block.filters li > span").bind("click", (event) => {
        //     event.preventDefault();
        //     let btn = $(event.currentTarget).closest("li");
        //     btn.toggleClass("active");
        //     let classes = "";
        //     $(".project-datasets-block.filters li.active").each((index, item) => {
        //         classes = `${classes}.${item.dataset.name}`;
        //     });
        //     let datasets = $(`.dataset-card-item`);
        //     datasets.removeClass("hidden");
        //     if (classes) datasets.not(classes).addClass("hidden");
        // });
        //
        // $(".dataset-card").bind("click", (event) => {
        //     event.preventDefault();
        //     if (!_isLoaded) return null;
        //     console.log(window.TerraProject.dataset);
        //     $(".params-container .params-item > .inner > .actions-form > .prepare > button").removeAttr("disabled");
        //     let btn = $(event.currentTarget).closest(".dataset-card-item");
        //     if (btn.hasClass("active")) return;
        //     $(".dataset-card-item").removeClass("active");
        //     btn.addClass("active");
        //     window.StatusBar.clear();
        //     window.TerraProject.dataset = event.currentTarget.dataset.name;
        //     let select_tasks = $("#dataset-task");
        //     select_tasks.html("");
        //     let tasks = window.TerraProject.datasets[window.TerraProject.dataset].tasks;
        //     for (let i=0; i<tasks.length; i++) {
        //         select_tasks.append($(`<option value="${tasks[i]}">${tasks[i]}</option>`));
        //     }
        //     if (select_tasks.children("option").length) {
        //         select_tasks.removeAttr("disabled");
        //     } else {
        //         select_tasks.attr("disabled", "disabled");
        //     }
        //     select_tasks.selectmenu("refresh");
        //     window.TerraProject.task = select_tasks.val();
        //     window.StatusBar.message(`Выбран датасет ${window.TerraProject.dataset}[${window.TerraProject.task}]`, true);
        // });
        //
        // let datasetTaskSelect = $("#dataset-task");
        // datasetTaskSelect.bind("change", (event) => {
        //     event.preventDefault();
        //     window.StatusBar.clear();
        //     window.TerraProject.task = event.currentTarget.value;
        //     window.StatusBar.message(`Выбран датасет ${window.TerraProject.dataset}[${window.TerraProject.task}]`, true);
        // }).selectmenu({
        //     change:(event) => {
        //         $(event.target).trigger("change");
        //     }
        // });
        //
        // $("main > .container > .properties .params").bind("submit", (event) => {
        //     event.preventDefault();
        //     if (!_isLoaded) return null;
        //     window.StatusBar.clear();
        //     if (!window.TerraProject.dataset) {
        //         window.StatusBar.message("Необходимо выбрать датасет", false);
        //         return;
        //     }
        //     if (!window.TerraProject.task) {
        //         window.StatusBar.message("Необходимо выбрать задачу", false);
        //         return;
        //     }
        //     window.ExchangeRequest(
        //         "prepare_dataset",
        //         (success, data) => {
        //             if (success) {
        //                 window.TerraProject.dataset = data.data.dataset;
        //                 window.TerraProject.task = data.data.task;
        //             } else {
        //                 window.StatusBar.message(data.error, false);
        //             }
        //         },
        //         {
        //             dataset:window.TerraProject.dataset,
        //             task:window.TerraProject.task
        //         }
        //     );
        //     window.ExchangeRequest(
        //         "get_data",
        //         (success, data) => {
        //             if (!success) {
        //                 window.StatusBar.message(data.error, false);
        //             } else {
        //                 _isLoaded = data.stop_flag;
        //                 // LockedForm(!_isLoaded);
        //                 window.StatusBar.message(data.data.status_string);
        //                 window.StatusBar.progress(data.data.progress_status.percents, data.data.progress_status.progress_text);
        //                 if (_isLoaded) {
        //                     window.StatusBar.message(`Загрузка датасета «${window.TerraProject.dataset} [${window.TerraProject.task}]» завершена`, true);
        //                     window.StatusBar.progress_clear();
        //                 }
        //             }
        //         }
        //     );
        // });
        //
        // if (window.TerraProject.dataset) {
        //     $(`.dataset-card[data-name=${window.TerraProject.dataset}]`).trigger("click");
        //     if (window.TerraProject.task) datasetTaskSelect.val(window.TerraProject.task).selectmenu("refresh");
        // }

        // $(window).bind("resize", onResize);
        // onResize();

    })


})(jQuery);
