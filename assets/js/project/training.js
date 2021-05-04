"use strict";


(($) => {


    const WAITING_FOR_THE_DATA = "Waiting for the data ...";

    let data_current_format = {
        'plots': [
            [
                {'x': [3, 6, 9, 12, 16, 18], 'y': [10, 20, 10, 30, 25, 30], 'name': 'label', 'mode': 'lines'},
                {'x': [3, 6, 9, 12, 16, 18], 'y': [30, 40, 20, 35, 30, 35], 'name': 'label2', 'mode': 'lines'},
            ],
            [
                {'x': [3, 6, 9, 12, 16, 18], 'y': [10, 20, 10, 30, 25, 35], 'name': 'label', 'mode': 'lines'},
                {'x': [3, 6, 9, 12, 16, 18], 'y': [30, 40, 20, 35, 30, 35], 'name': 'label2', 'mode': 'lines'},
            ],
        ],
        'scatters': [
            [
                {'x': [1, 5, 7, 2, 9, 3], 'y': [10, 40, 20, 50, 30, 60], 'name': 'label', 'mode': 'markers'},
                {'x': [1, 6, 3, 8, 4, 5], 'y': [10, 80, 20, 70, 30, 15], 'name': 'label2', 'mode': 'markers'}
            ]
        ]
    };

    let data_needed_format = {
        'plots': [
            {
                'list': [
                    {'x': [0, 1, 2], 'y': [0.0639, 0.0263, 0.0193], 'name': 'loss', 'mode': 'lines'},
                    {'x': [0, 1, 2], 'y': [0.0248, 0.0173, 0.0159], 'name': 'val_loss', 'mode': 'lines'}
                ],
                'title': 'loss and val_loss Epoch №003',
                'xaxis': {'title': 'epoch'},
                'yaxis': {'title': 'loss'}
            },
            {
                'list': [
                    {'x': [0, 1, 2], 'y': [0.978, 0.991, 0.994], 'name': 'accuracy', 'mode': 'lines'},
                    {'x': [0, 1, 2], 'y': [0.993, 0.995, 0.996], 'name': 'val_accuracy', 'mode': 'lines'}
                ],
                'title': 'accuracy and val_accuracy Epoch №003',
                'xaxis': {'title': 'epoch'},
                'yaxis': {'title': 'accuracy'}
            },
            {
                'list': [
                    {'x': [0, 1, 2], 'y': [1.005, 1.005, 1.005], 'name': 'val_accuracy class 0', 'mode': 'lines',},
                    {'x': [0, 1, 2], 'y': [1.031, 1.008, 1.008], 'name': 'val_accuracy class 1', 'mode': 'lines'},
                    {'x': [0, 1, 2], 'y': [1.025, 2, 2.25], 'name': 'val_accuracy class 2', 'mode': 'lines'},
                    {'x': [0, 1, 2], 'y': [1.55, 1.801, 1.902], 'name': 'val_accuracy class 3', 'mode': 'lines'},
                    {'x': [0, 1, 2], 'y': [1.235, 1.805, 2.123], 'name': 'val_accuracy class 4', 'mode': 'lines'},
                    {'x': [0, 1, 2], 'y': [1.335, 1.815, 2.213], 'name': 'val_accuracy class 5', 'mode': 'lines'},
                    {'x': [0, 1, 2], 'y': [1.435, 1.825, 2.343], 'name': 'val_accuracy class 6', 'mode': 'lines'},
                    {'x': [0, 1, 2], 'y': [1.535, 1.835, 2.453], 'name': 'val_accuracy class 7', 'mode': 'lines'},
                    {'x': [0, 1, 2], 'y': [1.635, 1.845, 2.563], 'name': 'val_accuracy class 8', 'mode': 'lines'},
                    {'x': [0, 1, 2], 'y': [1.735, 1.855, 2.673], 'name': 'val_accuracy class 9', 'mode': 'lines'}
                ],
                'title': 'val_accuracy of 10 classes. Epoch №003',
                'xaxis': {'title': 'epoch'},
                'yaxis': {'title': 'val_accuracy'}
            }
        ],
        'texts': "num_classes = 2, shape = (1799, 54, 96, 3), epochs = 20,\nlearning_rate=0.001, callbacks = [], batch_size = 32,\nshuffle = True, loss = categorical_crossentropy, metrics = ['accuracy']\n",
        'prints': [
            'Epoch 000 - loss:  0.7809 - accuracy:  0.7360 - val_loss:  0.5165 - val_accuracy:  0.7711',
            'Epoch 001 - loss:  0.4146 - accuracy:  0.8305 - val_loss:  0.5224 - val_accuracy:  0.7756',
            'Epoch 002 - loss:  0.3307 - accuracy:  0.8733 - val_loss:  0.5455 - val_accuracy:  0.7400',
            'Epoch 003 - loss:  0.2803 - accuracy:  0.8949 - val_loss:  0.5395 - val_accuracy:  0.7556',
            'Epoch 004 - loss:  0.2096 - accuracy:  0.9377 - val_loss:  0.5803 - val_accuracy:  0.7311',
            'Epoch 005 - loss:  0.1470 - accuracy:  0.9655 - val_loss:  0.6273 - val_accuracy:  0.7378',
            'Epoch 006 - loss:  0.1060 - accuracy:  0.9839 - val_loss:  0.7187 - val_accuracy:  0.7533',
            'Epoch 007 - loss:  0.0764 - accuracy:  0.9928 - val_loss:  0.7409 - val_accuracy:  0.7156',
            'Epoch 008 - loss:  0.0500 - accuracy:  0.9994 - val_loss:  0.7965 - val_accuracy:  0.7044',
            'Epoch 009 - loss:  0.0334 - accuracy:  1.0000 - val_loss:  0.8457 - val_accuracy:  0.7133',
            'Epoch 010 - loss:  0.0249 - accuracy:  1.0000 - val_loss:  0.8849 - val_accuracy:  0.7200',
            'Epoch 011 - loss:  0.0185 - accuracy:  1.0000 - val_loss:  0.9416 - val_accuracy:  0.7089',
            'Epoch 012 - loss:  0.0144 - accuracy:  1.0000 - val_loss:  0.9711 - val_accuracy:  0.7067',
            'Epoch 013 - loss:  0.0119 - accuracy:  1.0000 - val_loss:  1.0328 - val_accuracy:  0.7089',
            'Epoch 014 - loss:  0.0091 - accuracy:  1.0000 - val_loss:  1.0564 - val_accuracy:  0.7089',
            'Epoch 015 - loss:  0.0074 - accuracy:  1.0000 - val_loss:  1.0879 - val_accuracy:  0.7133',
            'Epoch 016 - loss:  0.0059 - accuracy:  1.0000 - val_loss:  1.1140 - val_accuracy:  0.7111',
            'Epoch 017 - loss:  0.0051 - accuracy:  1.0000 - val_loss:  1.1380 - val_accuracy:  0.7111',
            'Epoch 018 - loss:  0.0044 - accuracy:  1.0000 - val_loss:  1.1756 - val_accuracy:  0.7044',
            'Epoch 019 - loss:  0.0037 - accuracy:  1.0000 - val_loss:  1.1979 - val_accuracy:  0.7089'
        ]
    }

    let requestXHR;


    let DrawGraph = (plotName, container, data_needed_format) => {

        for (let i = 0; i < data_needed_format[plotName].length; i++) {
            let div = document.createElement("div");
            div.className = "graph";
            div.innerHTML = `<div class="data-visualizer" style="width: 100%; height: 100%"></div>`;

            container.append(div);

            let layout = {
                // height: 200,
                // width: 350, // если включить на responsive сбивается
                margin: {
                    l: 60,
                    r: 150,
                    t: 30,
                    b: 50,
                    pad: 0,
                    autoexpand: false,
                },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: {color: "#A7BED3"},
                showlegend: true,
                title: data_needed_format[plotName][i]["title"],
                legend: {
                    // orientation: "h",
                    font: {
                        family: "Open Sans",
                        color: "#A7BED3",
                    }
                },
                boxmode: 'group',
                xaxis: {
                    title: data_needed_format[plotName][i]["xaxis"]["title"],
                    showgrid: true,
                    zeroline: true,
                    linecolor: "rgba(10, 0, 9, .4)",
                    gridcolor: "rgba(14, 22, 33,.8)",
                    gridwidth: 0.2,

                },
                yaxis: {
                    title: data_needed_format[plotName][i]["yaxis"]["title"],
                    showline: true,
                    linecolor: "rgba(10, 0, 9, .4)",
                    gridcolor: "rgba(14, 22, 33,.8)",
                    gridwidth: 0.2,
                }
            };

            let config = {
                displaylogo: false,
                responsive: true,
                displayModeBar: false,
            };

            Plotly.newPlot(
                div.lastElementChild, data_needed_format[plotName][i]["list"],
                layout,
                config
            );
        }
    }

    let DisplayText = (data, container)=>{
        let rows = data["texts"].split("\n");
        rows.forEach((row)=>{
            row !== "" ? container.append(`<div class="text-item">${row}</div>`) : null;
        });
    }


    let UpdateTrainingProgress = (data) => {
        let training = $(".training-progress .info").html("");
        for (let index in data) {
            training.append(`<div class="item">${data[index]}</div>`);
        }
        $(".training-progress").mCustomScrollbar("scrollTo", "bottom");
    }


    let ResetGraphics = () => {
        let content = $(".graphics > .wrapper > .tabs-content > .inner");
        $(".graphics > .wrapper > .tabs > ul > li").removeClass("active").addClass("disabled");
        content.find(".tabs-item.graphs .tab-container").html("");
        content.find(".tabs-item.scatters .tab-container").html("");
        content.find(".tabs-item.images .tab-container").html("");
        content.find(".tabs-item.text .tab-container").html("");
    }

    $(() => {

        if (!window.TerraProject.dataset || !window.TerraProject.task) {
            let warning = $("#modal-window-warning").ModalWindow({
                title:"Предупреждение!",
                width:300,
                height:174,
                noclose:true,
                callback:(data) => {
                    warning.children(".wrapper").append($(`
                        <p>Для обучения необходимо загрузить датасет.</p>
                        <p><a class="format-link" href="${window.TerraProject.path.datasets}">Загрузить датасет</a></p>
                    `));
                }
            });
            warning.open();
        } else if (!Object.keys(window.TerraProject.layers).length) {
            let warning = $("#modal-window-warning").ModalWindow({
                title:"Предупреждение!",
                width:300,
                height:174,
                noclose:true,
                callback:(data) => {
                    warning.children(".wrapper").append($(`
                        <p>Для обучения необходимо загрузить модель.</p>
                        <p><a class="format-link" href="${window.TerraProject.path.modeling}">Загрузить модель</a></p>
                    `));
                }
            });
            warning.open();
        }

        $(".graphics > .wrapper > .tabs > ul > li > span").bind("click", (event) => {
            if ($(event.currentTarget).parent().hasClass("disabled")) return null;
            let item = $(event.currentTarget).parent(),
                content = $(".graphics > .wrapper > .tabs-content > .inner .tabs-item");
            item.parent().children("li").removeClass("active");
            item.addClass("active");
            content.removeClass("active");
            $(content[item.index()]).addClass("active");
        });

        $(".callback-params-block > .params-item > .inner > .actions-form > .evaluate > button").bind("click", (event) => {
            $(".callback-params-block > .params-item > .inner > .actions-form > .item > button").attr("disabled", "disabled");
            window.ExchangeRequest(
                "start_evaluate",
                (success, data) => {
                    if (success) {
                        $(".callback-params-block > .params-item > .inner > .actions-form > .training > button").removeAttr("disabled");
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                }
            );
            return false;
        });

        $(".callback-params-block > .params-item > .inner > .actions-form").bind("click", (event) => {
            event.preventDefault();
            window.StatusBar.clear();
            UpdateTrainingProgress([WAITING_FOR_THE_DATA]);
            ResetGraphics();
            window.ExchangeRequest(
                "start_nn_train",
                (success, data) => {
                    if (success) {
                        console.log("Training complete:", data);
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
                {
                    batch:$("#batch-size").val(),
                    epoch:$("#epoch-num").val(),
                    learning_rate:$("#field_form-learning_rate").val()
                }
            );
            window.ExchangeRequest(
                "get_data",
                (success, data) => {
                    if (success) {
                        if (!data.data.prints.length) data.data.prints = [WAITING_FOR_THE_DATA]
                        window.StatusBar.message(data.data.status_string);
                        window.StatusBar.progress(data.data.progress_status.percents, data.data.progress_status.progress_text);
                        $(".graphics > .wrapper > .tabs-content > .inner > .tabs-item .tab-container").html("");
                        DrawGraph("plots", $(".graphics .tabs-item.graphs > .tab-container"), data.data); // нарисовать для линейного
                        DrawGraph("scatters", $(".graphics .tabs-item.scatters > .tab-container"), data.data); // нарисовать для скаттера
                        DisplayText(data_needed_format, $(".graphics .tabs-item.text .tab-container")); // вывод текста
                        UpdateTrainingProgress(data.data.prints);
                        if (data.data.plots.length) $(".graphics > .wrapper > .tabs > ul > li.graphs").removeClass("disabled");
                        if (data.data.images.length) $(".graphics > .wrapper > .tabs > ul > li.images").removeClass("disabled");
                        if (data.data.texts) $(".graphics > .wrapper > .tabs > ul > li.text").removeClass("disabled");
                        if (data.data.scatters.length) $(".graphics > .wrapper > .tabs > ul > li.scatters").removeClass("disabled");
                        if (data.data.plots.length || data.data.images.length || data.data.texts || data.data.scatters.length) {
                            if (!$(".graphics > .wrapper > .tabs > ul > li.active").length) {
                                $(".graphics > .wrapper > .tabs > ul > li").not(".disabled").first().children("span").trigger("click");
                            }
                        }
                        if (data.data.stop_flag) {
                            $(".callback-params-block > .params-item > .inner > .actions-form > .training > button").attr("disabled", "disabled");
                            $(".callback-params-block > .params-item > .inner > .actions-form > .evaluate > button").removeAttr("disabled");
                        }
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                }
            );
        });

        let optimazerSelect = $("#optimazer");
        optimazerSelect.bind("change", (event) => {
            event.preventDefault();
            event.currentTarget.value
                && window.TerraProject.dataset !== null
                && window.TerraProject.model_name !== null
                && window.TerraProject.task !== null
                ? $(".callback-params-block > .params-item > .inner > .actions-form > .training > button").removeAttr("disabled")
                : $(".callback-params-block > .params-item > .inner > .actions-form > .training > button").attr("disabled", "disabled");
            $(".wrapper .params-optimazer-block .optimazer-item").html("");

            $(".wrapper .params-optimazer-block .optimazer-item").append(`
                <div class="inner form-inline-label inner-col-0"></div>
            `);
            window.ExchangeRequest(
                "get_optimizer_kwargs",
                (success, data) => {
                    if (success) {
                        let dataLen = Object.keys(data.data).length;
                        let dataEntries = Object.entries(data.data);
                        let column = Math.ceil(dataLen / 2)
                        // $(".params-optimazer-block .params-item").append(`
                        //     <div class="inner form-inline-label inner-col-1"></div>
                        // `);
                        dataEntries.forEach(([key, param], index) => {
                            let is_boolean = param.type === 'bool';
                            let widget = window.FormWidget(key, {
                                list: param.list,
                                available: param.available,
                                type: param.type,
                                default: param.value.toString(),
                                name: key,
                                checkbox: is_boolean,
                            });
                            widget.addClass("field-inline");

                            // if (index < column && dataLen > 4) {
                            //     $(".params-optimazer-block .inner.inner-col-0").append(widget);
                            // } else {
                            //     $(".params-optimazer-block .inner.inner-col-1").append(widget);
                            // }
                            $(".params-optimazer-block .inner.inner-col-0").append(widget);
                        });
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
                {"optimizer":$(event.currentTarget).val()}
            )
        }).selectmenu({
            change:(event) => {
                $(event.target).trigger("change");
            }
        }).trigger("change");

        $(".callbacks-switchers > .field-inline > .checkout-switch > input").bind("change", (event) => {
            event.preventDefault();
            let send_data = {};
            $(".callbacks-switchers input").each((index, item) => {
                send_data[item.name] = item.checked;
            });
            window.ExchangeRequest(
                "set_callbacks_switches",
                (success, data) => {
                    if (success) {
                        window.TerraProject.callbacks = data.data.callbacks;
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
                send_data
            )
        });

        $(".task-select").bind("change", (event) => {
            event.preventDefault();

        }).selectmenu({
            change:(event) => {
                $(event.target).trigger("change");
                let loss = $("#"+event.target.parentNode.parentNode.id).find(".loss-select");
                let metric = $("#"+event.target.parentNode.parentNode.id).find(".metric-select");
                loss.empty();
                metric.empty();

                window.TerraProject.compile[event.target.value].losses.forEach((elem) => {
                    loss.append(`<option value="${elem}">${elem}</option>`);
                });

                 window.TerraProject.compile[event.target.value].metrics.forEach((elem) => {
                    metric.append(`<option value="${elem}">${elem}</option>`);
                });

                loss.selectmenu("refresh");
                metric.selectmenu("refresh");

                // $("#"+event.target.parentNode.parentNode.id).find(".loss-select").refresh();
            }
        }).trigger("change");


        $(".click-menu").bind("click", (event)=>{
            $(event.target).parent().toggleClass('open');
        });

        $(".item.training > button").bind("click", (event)=>{
            event.preventDefault();
            let form = $(event.currentTarget),
                serializeData = form.parents('.params-container').serializeArray();

            console.log(serializeData)
        });

        // здесь будет проверка на наличие флага "регрессия"
        // if (data_needed_format["scatters"]){
        //     let tabLink = document.createElement("li");
        //     tabLink.className = "params-link";
        //     tabLink.innerHTML = `<a href="#tabs-4">Скаттеры</a>`;
        //     document.getElementById("graphsParamsTabs").append(tabLink);
        //
        //     let graphElem = document.createElement("div");
        //     graphElem.className = "graph-elem-block custom-scrollbar-wrapper graph-scatter";
        //     graphElem.id = "tabs-4";
        //     graphElem.innerHTML = `<div class="container data-container" id="scattersContainer"></div>`;
        //     document.getElementById("tabs").append(graphElem);
        // }

        DrawGraph("plots", $(".tab-container"), data_needed_format); // нарисовать для линейного
        // DrawGraph("scatters", $(".graphics .tabs-item.scatters > .tab-container")); // нарисовать для скаттера

        // DrawGraph("plots", $(".tab-container"), data_needed_format);

    });


})(jQuery);
