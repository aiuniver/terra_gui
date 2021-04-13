"use strict";


(($) => {


    let _isLoaded = true;


    let onResize = () => {
        let datasets = $(".project-datasets-block.datasets");
        datasets.css("padding-top", `${$(".project-datasets-block.filters").innerHeight()+1}px`);
        datasets.removeClass("hidden");
    }


    let LockedForm = (locked) => {
        let button = $(".params-container .params-item > .inner > .actions-form > .prepare > button"),
            select = $("#dataset-task"),
            datasets = $(".dataset-card-item").not(".active").children(".dataset-card");
        if (locked) {
            button.attr("disabled", "disabled");
            select.attr("disabled", "disabled").selectmenu("refresh");
            datasets.addClass("disabled");
        } else {
            button.removeAttr("disabled");
            select.removeAttr("disabled").selectmenu("refresh");
            datasets.removeClass("disabled");
        }
    }


    $(() => {

        $(".project-datasets-block.filters li > span").bind("click", (event) => {
            event.preventDefault();
            let btn = $(event.currentTarget).closest("li");
            btn.toggleClass("active");
            let classes = "";
            $(".project-datasets-block.filters li.active").each((index, item) => {
                classes = `${classes}.${item.dataset.name}`;
            });
            let datasets = $(`.dataset-card-item`);
            datasets.removeClass("hidden");
            if (classes) datasets.not(classes).addClass("hidden");
        });

        $(".dataset-card").bind("click", (event) => {
            event.preventDefault();
            if (!_isLoaded) return null;
            let btn = $(event.currentTarget).closest(".dataset-card-item");
            if (btn.hasClass("active")) return;
            $(".dataset-card-item").removeClass("active");
            btn.addClass("active");
            window.StatusBar.clear();
            window.TerraProject.dataset = event.currentTarget.dataset.name;
            let select_tasks = $("#dataset-task");
            select_tasks.html("");
            let tasks = window.TerraProject.datasets[window.TerraProject.dataset].tasks;
            for (let i=0; i<tasks.length; i++) {
                select_tasks.append($(`<option value="${tasks[i]}">${tasks[i]}</option>`));
            }
            if (select_tasks.children("option").length) {
                select_tasks.removeAttr("disabled");
            } else {
                select_tasks.attr("disabled", "disabled");
            }
            select_tasks.selectmenu("refresh");
            window.TerraProject.task = select_tasks.val();
            window.StatusBar.message(`Выбран датасет ${window.TerraProject.dataset}[${window.TerraProject.task}]`, true);
        });

        let datasetTaskSelect = $("#dataset-task");
        datasetTaskSelect.bind("change", (event) => {
            event.preventDefault();
            window.StatusBar.clear();
            window.TerraProject.task = event.currentTarget.value;
            window.StatusBar.message(`Выбран датасет ${window.TerraProject.dataset}[${window.TerraProject.task}]`, true);
        }).selectmenu({
            change:(event) => {
                $(event.target).trigger("change");
            }
        });

        $("main > .container > .properties .params").bind("submit", (event) => {
            event.preventDefault();
            if (!_isLoaded) return null;
            window.StatusBar.clear();
            if (!window.TerraProject.dataset) {
                window.StatusBar.message("Необходимо выбрать датасет", false);
                return;
            }
            if (!window.TerraProject.task) {
                window.StatusBar.message("Необходимо выбрать задачу", false);
                return;
            }
            window.ExchangeRequest(
                "prepare_dataset",
                (success, data) => {
                    if (success) {
                        window.TerraProject.dataset = data.data.dataset;
                        window.TerraProject.task = data.data.task;
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
                {
                    dataset:window.TerraProject.dataset,
                    task:window.TerraProject.task
                }
            );
            window.ExchangeRequest(
                "get_data",
                (success, data) => {
                    if (!success) {
                        window.StatusBar.message(data.error, false);
                    } else {
                        _isLoaded = data.stop_flag;
                        LockedForm(!_isLoaded);
                        window.StatusBar.message(data.data.status_string);
                        window.StatusBar.progress(data.data.progress_status.percents, data.data.progress_status.progress_text);
                        if (_isLoaded) {
                            window.StatusBar.message(`Загрузка датасета «${window.TerraProject.dataset} [${window.TerraProject.task}]» завершена`, true);
                            window.StatusBar.progress_clear();
                        }
                    }
                }
            );
        });

        if (window.TerraProject.dataset) {
            $(`.dataset-card[data-name=${window.TerraProject.dataset}]`).trigger("click");
            if (window.TerraProject.task) datasetTaskSelect.val(window.TerraProject.task).selectmenu("refresh");
        }

        $(window).bind("resize", onResize);
        onResize();

    })


})(jQuery);
