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
                _cnodes = _d3graph.select("#canvas-nodes"),
                svg = $(_d3graph._groups[0][0]),
                _onContextDrag = false,
                _onDrag = false,
                _sourceNode,
                _targetNode;

            let __clear = () => {
                _clines.selectAll("g").remove();
                _cnodes.selectAll("g").remove();
            }

            let _create_node = (layer) => {

                let w = _d3graph._groups[0][0].width.baseVal.value,
                    h = _d3graph._groups[0][0].height.baseVal.value;

                layer.index = $(".node").length + 1;
                layer.config.name = $(".node").length + 1;

                let node = _cnodes.append("g")
                    .attr("id", `node-${layer.index}`)
                    .attr("class", `node node-type-${layer.config.type}`)
                    .call(d3.drag()
                        .on("start", _node_dragstarted)
                        .on("drag", _node_dragged)
                        .on("end", _node_dragended)
                    );

                let rect = node.append("rect");

                let text = node.append("text")
                    .text(`${layer.config.name}: ${layer.config.type}`)
                    .attr("x", 10)
                    .attr("y", 17);

                let width = text._groups[0][0].getBBox().width + 20;
                    rect.attr("width", width);

                if (layer.x === undefined ) layer.x = 30;
                if (layer.y === undefined) layer.y = 30;

                let target_circle = node.append("circle")
                    .attr("class", "dot-target")
                    .attr("cx", width/2)
                    .attr("cy", -4);

                let source_circle = node.append("circle")
                    .attr("class", "dot-source")
                    .attr("cx", width/2)
                    .attr("r", 5)
                    .attr("cy", _LINE_HEIGHT);

                node.data([layer])
                    .attr("transform", "translate(" + layer.x + "," + layer.y + ")");

                $(".node").bind("mousedown", _onmousedown)
                    .bind("mouseup", _onmouseup);
            };

            let _create_line = (layer) => {

                let _source_node_point = {x:_sourceNode.transform.baseVal[0].matrix.e, y: _sourceNode.transform.baseVal[0].matrix.f};

                let line = _clines.append("line")
                    .attr("id", `line-${$(".line").length + 1}`)
                    .attr("class", "line")
                    .attr("x1", _source_node_point.x + _sourceNode.children[0].width.baseVal.value/2)
                    .attr("y1", _source_node_point.y + _LINE_HEIGHT)
                    .attr("x2", 30)
                    .attr("y2", 30);

                let node_data = _cnodes.select("#" + _sourceNode.id).data()[0];

                node_data.lineSource.push(line);
                _cnodes.select("#" + _sourceNode.id).data(node_data);
            };

            let _change_line = () => {
                 let _target_node_point = {x:_targetNode.transform.baseVal[0].matrix.e, y: _targetNode.transform.baseVal[0].matrix.f};

                 let line = _clines.select("#line-" + $(".line").length);

                 line.attr("x2", _target_node_point.x + _targetNode.children[0].width.baseVal.value/2);
                 line.attr("y2", _target_node_point.y - 4);

                 let next_node_data = _cnodes.select("#" + _targetNode.id).data()[0];

                 line.data([{source:  _cnodes.select("#" + _sourceNode.id), target:  _cnodes.select("#" + _targetNode.id)}]);

                 next_node_data.lineTarget.push(line) ;
                 _cnodes.select("#" + _targetNode.id).data(next_node_data);
            };


            this.activeNode = (_node) => {
                _cnodes.selectAll(".node").classed("active", false);
                _node.classed("active", true);
                // params.load(_node.data()[0]);
            }


            let _node_dragstarted = (data) => {
                terra_board.find(".canvas > .hint").remove();
                let _node = d3.select(`#node-${data.index}`);
                _node.raise().classed("hover", true);
            }

            let _node_dragged = (data) => {
                _onDrag = true;
                let _node = d3.select(`#node-${data.index}`);
                 _node.attr("transform", () => {
                     data.x = d3.event.x;
                     data.y = d3.event.y;
                     return "translate(" + d3.event.x + "," + d3.event.y + ")";
                 });

                 //DONT WORK

                 let _node_data = _node.data()[0],
                     lineTarget = _node_data.lineTarget[0]._groups[0],
                     lineSource = _node_data.lineSource[0]._groups[0];

                 if (lineTarget) {
                     for(let i in lineTarget) {
                         let cx =  d3.event.x + _node.select("rect")._groups[0][0].width.baseVal.value / 2,
                             cy = d3.event.y;
                             // ox = lineTarget[i]._groups[0][0].transform.baseVal[0].matrix.e,
                             // oy = lineTarget[i]._groups[0][0].transform.baseVal[0].matrix.f;
                         // lineTarget[i].select(".dot-target")
                         //     .attr("cx", cx - ox - 1)
                         //     .attr("cy", cy - oy - 4);
                         _clines.select($(lineTarget[i][0].id))
                             .attr("x2", cx)
                             .attr("y2", cy - 4);
                     }
                 }
                 if (lineSource) {
                     for(let i in lineSource) {
                         let cx = d3.event.x  + _node.select("rect")._groups[0][0].width.baseVal.value / 2,
                             cy = d3.event.y  + _NODE_HEIGHT;
                         //     ox = lineSource[i]._groups[0][0].transform.baseVal[0].matrix.e,
                         //     oy = lineSource[i]._groups[0][0].transform.baseVal[0].matrix.f;
                         // lineSource[i].select(".dot-source")
                         //     .attr("cx", cx - ox - 1)
                         //     .attr("cy", cy - oy + 4);
                         lineSource[i].select("line")
                             .attr("x1", cx)
                             .attr("y1", cy + 4);
                     }
                 }
            }

            let _node_dragended = (data) => {
                let _node = d3.select(`#node-${data.index}`);
                _node.classed("hover", false);
                if (!_onDrag) this.activeNode(_node);
                _onDrag = false;
            };

            $(".canvas-container").bind("contextmenu", (event) => {
                return false;
            });

            let _onmousedown = (event)=>{
                svg.bind("mousemove", _onmousemove);
                _sourceNode = event.target.parentNode;
                _targetNode = undefined;
                _create_line();
            };

            let _onmouseup = (event)=>{
                svg.unbind("mousemove", _onmousemove);
                _targetNode = event.target.parentNode;
                _onContextDrag = false;
                _change_line();
            };

            let _onmousemove = (event)=>{
                _onContextDrag = true;

                d3.select("#line-" + $(".line").length)
                    .attr("x2", event.offsetX)
                    .attr("y2", event.offsetY);
            };

            $(document).bind("keydown", (event) => {
                if(event.which == 187){
                    let layer = {
                        config: middle_cfg,
                        lineTarget: [],
                        lineSource: []
                    }
                    _create_node(layer);
                }
            })

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


    });


    $(() => {

        terra_toolbar = $(".project-modeling-toolbar").TerraToolbar();
        terra_board = $(".canvas-container").TerraBoard();
        terra_params = $(".params-container").TerraParams();

        if (!window.TerraProject.dataset || !window.TerraProject.task) {
            // let warning = $("#modal-window-warning").ModalWindow({
            //     title:"Предупреждение!",
            //     width:300,
            //     height:174,
            //     noclose:true,
            //     callback:(data) => {
            //         warning.children(".wrapper").append($(`
            //             <p>Для редактирования модели необходимо загрузить датасет.</p>
            //             <p><a class="format-link" href="${window.TerraProject.path.datasets}">Загрузить датасет</a></p>
            //         `));
            //     }
            // });
            // warning.open();
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

    });

    let middle_cfg = {
        type: "middle",
        params: {}
    }

})(jQuery);

