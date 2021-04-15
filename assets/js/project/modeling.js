"use strict";


(($) => {


    let terra_toolbar, terra_board, terra_params;


    let LoadModel = $("#modal-window-load-model").ModalWindow({
        title:window.Messages.get("LOAD_MODEL"),
        width:680,
        height:440,
        request:["get_models"]
    });


    $.fn.extend({


        TerraToolbar: function() {

            if (!this.length) return this;

            let _execute = {
                load: () => {
                    LoadModel.open(
                        (block, data) => {
                            block.find(".models-data > .models-list .loaded-list").html("");
                            block.find(".models-data > .model-arch > .wrapper").addClass("hidden");
                            for (let index in data) {
                                block.find(".models-data > .models-list .loaded-list").append($(`<li data-name="${data[index]}"><span>${data[index]}</span></li>`))
                            }
                            block.find(".models-data > .models-list .loaded-list > li > span").bind("click", (event) => {
                                let item = $(event.currentTarget).parent();
                                item.parent().children("li").removeClass("active");
                                item.addClass("active");
                                block.find(".models-data > .model-arch > .wrapper").addClass("hidden");
                                window.StatusBar.message(window.Messages.get("TRYING_TO_LOAD_MODEL", [item.data("name")]));
                                window.ExchangeRequest(
                                    "get_model_from_list",
                                    (success, data) => {
                                        if (success) {
                                            window.StatusBar.message(window.Messages.get("MODEL_LOADED", [item.data("name")]), true);
                                            block.find(".models-data > .model-arch > .wrapper > .model-arch-img > img").attr("src", `data:image/png;base64,${data.data.image}`);
                                            block.find(".models-data > .model-arch > .wrapper").removeClass("hidden");
                                            block.find(".models-data > .model-arch > .wrapper > .model-save-arch-btn > button")[0].ModelData = data.data;
                                        } else {
                                            window.StatusBar.message(data.error, false);
                                        }
                                    },
                                    {"model_file":item.data("name")}
                                );
                            });
                        }
                    );
                },
                save: (item, callback) => {
                    if (typeof callback === "function") callback(item);
                },
                input: (item, callback) => {
                    if (typeof callback === "function") callback(item);
                },
                middle: (item, callback) => {
                    if (typeof callback === "function") callback(item);
                },
                output: (item, callback) => {
                    if (typeof callback === "function") callback(item);
                }
            }

            this.layersReset = (input, middle, output) => {
                if (!input) {
                    input = false;
                    middle = true;
                    output = true;
                } else if (!output) {
                    input = true;
                    middle = true;
                    output = false;
                } else {
                    input = true;
                    middle = false;
                    output = true;
                }
                this.find(".menu-section.layers > li[data-type=input]")[0].disabled = input;
                this.find(".menu-section.layers > li[data-type=middle]")[0].disabled = middle;
                this.find(".menu-section.layers > li[data-type=output]")[0].disabled = output;
            }

            Object.defineProperty(this, "items", {
                get: () => {
                    return this.find(".menu-section > li");
                }
            });

            this.items.each((index, item) => {
                Object.defineProperty(item, "disabled", {
                    set: (value) => {
                        value
                            ? item.setAttribute("disabled", "disabled")
                            : item.removeAttribute("disabled");
                    },
                    get: () => {
                        return item.hasAttribute("disabled");
                    }
                });
                item.execute = (callback) => {
                    let _method = _execute[item.dataset.type];
                    if (item.disabled) return;
                    if (typeof _method !== "function") {
                        item.disabled = true;
                    } else {
                        _method(item, callback);
                    }
                }
            });

            return this;

        },


        TerraBoard: function() {

            if (!this.length) return this;

            const _NODE_HEIGHT = 25,
                _LINE_HEIGHT = 30;

            let _d3graph = d3.select(this.find(".canvas > svg")[0]),
                _clines = _d3graph.select("#canvas-lines"),
                _cnodes = _d3graph.select("#canvas-nodes");

            let __clear = () => {
                _clines.selectAll("g").remove();
                _cnodes.selectAll("g").remove();
            }

            Object.defineProperty(this, "model", {
                set: (layers) => {
                    __clear();
                    // let num = 0,
                    //     _layer,
                    //     _layers = [];
                    // for (let index in value) {
                    //     let type = "middle";
                    //     if (num === Object.keys(value).length - 1) type = "output";
                    //     if (num === 0) type = "input";
                    //     _layer = {
                    //         index:index,
                    //         config:value[index],
                    //         type:type
                    //     };
                    //     _layers.push(_layer);
                    //     this.layer = _layer
                    //     num++;
                    // }
                    // _create_model(_layers);
                    // let exists = _existsLayersTypes();
                    // toolbar.layersReset(exists[0], exists[1], exists[2]);
                    // params.reset();
                },
                get: () => {
                    return _cnodes.selectAll("g.node");
                }
            });

            return this;

        },


        TerraParams: function() {

            if (!this.length) return this;

            return this;

        }


    })


    $(() => {

        terra_toolbar = $(".project-modeling-toolbar").TerraToolbar();
        terra_board = $(".canvas-container").TerraBoard();
        terra_params = $(".params-container").TerraParams();

        if (!window.TerraProject.dataset || !window.TerraProject.task) {
            let warning = $("#modal-window-warning").ModalWindow({
                title:"Предупреждение!",
                width:300,
                height:174,
                noclose:true,
                callback:(data) => {
                    warning.children(".wrapper").append($(`
                        <p>Для редактирования модели необходимо загрузить датасет.</p>
                        <p><a class="format-link" href="${window.TerraProject.path.datasets}">Загрузить датасет</a></p>
                    `));
                }
            });
            warning.open();
        } else {
            terra_board.model = window.TerraProject.layers;
        }

        LoadModel.find(".model-save-arch-btn > button").bind("click", (event) => {
            window.StatusBar.clear();
            window.ExchangeRequest(
                "set_model",
                (success, data) => {
                    if (success) {
                        me.model = data.data.layers;
                        LoadModel.close();
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
                {"layers":event.currentTarget.ModelData.layers}
            )
        });

        terra_toolbar.items.children("span").bind("click", (event) => {
            event.currentTarget.parentNode.execute((item) => {
                if ($(item.parentNode).hasClass("layers")) _loadLayer(item.dataset.type);
            });
        });

    })

})(jQuery);
