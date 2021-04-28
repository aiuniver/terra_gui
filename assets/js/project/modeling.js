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
                     let nodes = d3.selectAll("g.node").data(),
                         send_data = {};
                     for(let node in nodes){
                         delete nodes[node].lineSource;
                         delete nodes[node].lineTarget;
                         send_data[nodes[node].id] = nodes[node];
                     }
                     window.StatusBar.clear();
                     window.ExchangeRequest(
                         "set_model",
                         (success, data) => {
                             if (success) {
                                 this.btn.save.disabled = true;
                                 window.StatusBar.message(window.Messages.get("MODEL_SAVED"), true);
                                 if (typeof callback === "function") callback(item);
                             } else {
                                 window.StatusBar.message(data.error, false);
                             }
                         },
                         {"layers": send_data, "schema": []}
                     );
                },
                validation: (item, callback) => {
                    window.StatusBar.clear();
                    window.ExchangeRequest(
                        "get_change_validation",
                        (success, data) => {
                            if (success) {
                                console.log(data);
                            } else {
                                window.StatusBar.message(data.error, false);
                            }
                        }
                    );
                },
                input: (item, callback) => {
                    if (typeof callback === "function") callback(item);
                },
                middle: (item, callback) => {
                    if (typeof callback === "function") callback(item);
                },
                output: (item, callback) => {
                    if (typeof callback === "function") callback(item);
                },
                clear: (item, callback) => {
                    terra_board.clear();
                }
            }

            Object.defineProperty(this, "items", {
                get: () => {
                    return this.find(".menu-section > li");
                }
            });

            Object.defineProperty(this, "btn", {
                get: () => {
                    return {
                        "load":this.find(".menu-section > li[data-type=load]")[0],
                        "save":this.find(".menu-section > li[data-type=save]")[0],
                        "validation":this.find(".menu-section > li[data-type=validation]")[0],
                        "input":this.find(".menu-section > li[data-type=input]")[0],
                        "middle":this.find(".menu-section > li[data-type=middle]")[0],
                        "output":this.find(".menu-section > li[data-type=output]")[0],
                        "clear":this.find(".menu-section > li[data-type=clear]")[0],
                    };
                }
            });

            this.items.each((index, item) => {
                Object.defineProperty(item, "disabled", {
                    set: (value) => {
                        if (value) $(item).attr("disabled", "disabled");
                        else $(item).removeAttr("disabled");
                    },
                    get: () => {
                        return item.hasAttribute("disabled");
                    }
                });
                item.execute = (callback) => {
                    if (!item.disabled) {
                        let _method = _execute[item.dataset.type];
                        if (typeof _method == "function") _method(item, callback);
                    }
                }
            });

            return this;

        },


        TerraBoard: function() {

            if (!this.length) return this;

            const _NODE_HEIGHT = 26,
                _LINE_HEIGHT = 30;


            let zoom = d3.zoom().on("zoom", zoomed);

            let _d3graph = d3.select(this.find(".canvas > svg")[0]),
                _clines = _d3graph.select("#canvas-lines"),
                _cnodes = _d3graph.select("#canvas-nodes"),
                svg = $(_d3graph._groups[0][0]),
                _layer_row_w = [],
                _model_schema = [],
                _onContextDrag = false,
                _onDrag = false,
                _onNode,
                _new_link,
                _sourceNode,
                _targetNode,
                _lastNodeId = 0,
                _lastLineId = 0;

            _d3graph.call(zoom);  

            d3.select("#zoom-inc").on("click", () => {  
                zoom.scaleBy(_d3graph.transition().duration(450), 1.2);
            });
            
            d3.select("#zoom-dec").on("click", () => {
                zoom.scaleBy(_d3graph.transition().duration(450), 0.8);
            });

            d3.select("#zoom-reset").on("click", () => {
                _d3graph.transition().duration(450).call(zoom.transform, d3.zoomIdentity);
            });
                  
            function zoomed() {
                _d3graph.select("g").attr("transform", d3.event.transform);
            }

            this.load_layer = (class_name) => {
                let type;
                switch (class_name) {
                    case "input":
                        type = "Input";
                        break;
                    case "output":
                        type = "Dense";
                        break;
                    default:
                        type = "Conv2D";
                        break;
                }

                let _max_id = 0;
                for (let key in window.TerraProject.layers) {
                    if (window.TerraProject.layers[key].index > _max_id) _max_id = window.TerraProject.layers[key].index;
                }
                _max_id++;

                let layer_config = {
                    name: `l${_max_id}_${type}`,
                    input_shape: [],
                    output_shape: [],
                    params: $.extend(true, {}, window.TerraProject.layers_types[type]),
                    type: type,
                    up_link: []
                };

                for (let group in layer_config.params) {
                    for (let param in layer_config.params[group]) {
                        layer_config.params[group][param] = layer_config.params[group][param].default;
                    }
                }

                let layer_default = {
                    config: layer_config,
                    type: class_name,
                    id: _max_id,
                    index: _max_id,
                };

                _create_node(layer_default);
                window.TerraProject.layers[_max_id] = layer_default;
                this.activeNode(d3.select(`#node-${_max_id}`)._groups[0][0]);

                window.ExchangeRequest(
                    "save_layer",
                    null,
                    d3.select(`#node-${_max_id}`).data()[0]
                );
            };

            let _change_line = (new_line = false) => {
                let line_id = "line-" + _lastLineId;
                let _target_node_point = {x:_targetNode.transform.baseVal[0].matrix.e, y: _targetNode.transform.baseVal[0].matrix.f};

                let repeat_line = false,
                    cycle_line = false

                if(new_line){
                    let _target_node_d3 = d3.select("#"+_targetNode.id),
                        _target_node_d3_data = _target_node_d3.data(),
                        _source_node_d3 = d3.select("#"+_sourceNode.id),
                        _source_node_d3_data = _source_node_d3.data();

                    if(_target_node_d3_data[0].config.up_link.indexOf(_sourceNode.__data__.id) != -1) repeat_line = true;
                    if(_source_node_d3_data[0].config.up_link.indexOf(_targetNode.__data__.id) != -1) cycle_line = true;

                    _target_node_d3_data[0].config.up_link.push(_sourceNode.__data__.id);
                    _target_node_d3.data(_target_node_d3_data);

                }



                let line = _clines.select("#" + line_id);

                line.attr("x2", _target_node_point.x + _targetNode.children[0].width.baseVal.value/2);
                line.attr("y2", _target_node_point.y - 4);

                let next_node_data = _cnodes.select("#" + _targetNode.id).data()[0];

                line.data([{source:  _cnodes.select("#" + _sourceNode.id), target:  _cnodes.select("#" + _targetNode.id)}]);
                next_node_data.lineTarget[line_id] = line;
                _cnodes.select("#" + _targetNode.id).select(".dot-target").attr("visibility", "visible");
                _cnodes.select("#" + _targetNode.id).data(next_node_data);

                if(_targetNode.id == _sourceNode.id || repeat_line || cycle_line){
                    _delete_line($("#" + line_id)[0]);
                    _lastLineId--;
                }
            };


            this.activeNode = (g) => {
                let node = _cnodes.select(`#${g.id}`);
                _cnodes.selectAll(".node").classed("active", false);
                node.classed("active", true);
                terra_params.load(node.data()[0]);
            }

            this.bind("contextmenu", (event) => {
                return false;
            });

            $(document).bind("mousedown", (event) => {
                this.find(".canvas > .hint").remove();
            });

            $(document).bind("keyup", (event) => {
                if (event.keyCode === 27) {
                    this.removeClass("onlink");
                    if (_new_link) {
                        if (_onNode) _update_dots_per_node(_d3graph.select(`#${_onNode[0].id}`));
                        _update_dots_per_node(_d3graph.select(`#node-${_new_link._groups[0][0].sourceID}`));
                        _new_link.remove();
                        _new_link = undefined;
                    }
                    _clines.selectAll("line").classed("active", false);
                }
                if (event.keyCode === 46) {
                    let line_active = $("line.line.active");
                    if (line_active.length) {
                        _remove_line(line_active);
                        terra_toolbar.btn.save.disabled = false;
                        $(terra_toolbar.btn.save).children("span").trigger("click");
                    }
                }
            });

            let _remove_line = (line) => {
                let match = line[0].id.match(/^line_([\d]+)_([\d]+)$/);
                if (match.length === 3) {
                    let sourceID = parseInt(match[1]),
                        targetID = parseInt(match[2]),
                        sourceNode = _d3graph.select(`#node-${sourceID}`),
                        targetNode = _d3graph.select(`#node-${targetID}`),
                        sourceData = sourceNode.data()[0],
                        targetData = targetNode.data()[0];
                    sourceData.down_link = sourceData.down_link.filter((item) => {return item !== targetID});
                    targetData.config.up_link = targetData.config.up_link.filter((item) => {return item !== sourceID});
                    sourceNode.data([sourceData]);
                    targetNode.data([targetData]);
                    _update_dots_per_node(sourceNode);
                    _update_dots_per_node(targetNode);
                    line.remove();
                }
            }

            svg.bind("mousemove", (event) => {
                if (_new_link) {
                    let x2 = event.offsetX,
                        y2 = event.offsetY;
                    if (_onNode) {
                        if (`${_new_link._groups[0][0].sourceID}` !== `${_onNode[0].__data__.id}` && _onNode[0].__data__.config.up_link.indexOf(_new_link._groups[0][0].sourceID) === -1) {
                            let matrix = _onNode[0].transform.baseVal[0].matrix;
                            x2 = matrix.e;
                            y2 = matrix.f - _LINE_HEIGHT / 2 - 2;
                            _onNode.children(".dot-target").attr("visibility", "visible");
                        }
                    }
                    _new_link.attr("x2", x2).attr("y2", y2);
                }
            });

            let _node_dragstarted = (data, _, rect) => {
                let node = $(rect).parent()[0];
                // this.find(".canvas > .hint").remove();
                // let _node = d3.select(`#node-${data.id}`);
                // _node.raise().classed("hover", true);
                //  if (!_onDrag) this.activeNode(_node);
                //  _onDrag = false;
            }

            let _node_dragged = (data, _, rect) => {
                let node = $(rect).parent()[0],
                    _node = d3.select(`#${node.id}`),
                    info = _node.data()[0];
                _onDrag = true;
                _node.attr("transform", (data) => {
                    let transform = _d3graph.select("#canvas-container")._groups[0][0].transform,
                        zoom_value = transform.baseVal[1].matrix.a;
                    data.x = d3.event.sourceEvent.layerX - d3.event.subject.x;
                    data.y = d3.event.sourceEvent.layerY - d3.event.subject.y;
                    return `translate(${data.x},${data.y})`;
                }).raise();

                let matrix = node.transform.baseVal[0].matrix,
                    x = matrix.e, y = matrix.f;

                if (info.down_link === undefined) info.down_link = [];
                info.down_link.forEach((id) => {
                    _clines.select(`#line_${info.id}_${id}`).attr("x1", x).attr("y1", y+_LINE_HEIGHT/2+2);
                });
                info.config.up_link.forEach((id) => {
                    _clines.select(`#line_${id}_${info.id}`).attr("x2", x).attr("y2", y-_LINE_HEIGHT/2-2);
                });
                // console.log(_clines.select(`line[data-target=${info.config.up_link}]`));
                //
                //  let _node_data = _node.data()[0],
                //      lineTarget = _node_data.lineTarget,
                //      lineSource = _node_data.lineSource;
                //
                //  if (lineTarget) {
                //      for(let i in lineTarget) {
                //          let cx =  d3.event.x + _node.select("rect")._groups[0][0].width.baseVal.value / 2,
                //              cy = d3.event.y;
                //
                //          _clines.select("#"+lineTarget[i]._groups[0][0].id)
                //              .attr("x2", cx)
                //              .attr("y2", cy - 4);
                //      }
                //  }
                //  if (lineSource) {
                //      for(let i in lineSource) {
                //          let cx = d3.event.x  + _node.select("rect")._groups[0][0].width.baseVal.value / 2,
                //              cy = d3.event.y  + _NODE_HEIGHT;
                //
                //          _clines.select("#"+lineSource[i]._groups[0][0].id)
                //              .attr("x1", cx)
                //              .attr("y1", cy + 4);
                //      }
                //  }
            }

            let _node_dragended = (data, _, rect) => {
                let node = $(rect).parent()[0],
                    _node = d3.select(`#${node.id}`);
                if (_onDrag) {
                    window.ExchangeRequest(
                        "save_layer",
                        null,
                        _node.data()[0]
                    );
                } else {
                    if (this.hasClass("onlink")) {
                        let matrix = node.transform.baseVal[0].matrix,
                            sourceID = _new_link._groups[0][0].sourceID,
                            sourceNode = _d3graph.select(`#node-${sourceID}`),
                            sourceData = sourceNode.data()[0],
                            targetID = node.__data__.id,
                            targetNode = _d3graph.select(`#node-${targetID}`),
                            targetData = targetNode.data()[0];
                        if (`${sourceID}` !== `${targetID}` && targetData.config.up_link.indexOf(sourceID) === -1) {
                            this.removeClass("onlink");
                            _new_link.attr("id", `line_${sourceID}_${targetID}`)
                                .attr("x2", matrix.e)
                                .attr("y2", matrix.f - _LINE_HEIGHT / 2 - 2);
                            _new_link = undefined;
                            if (!sourceData.down_link) sourceData.down_link = [];
                            sourceData.down_link.push(targetID);
                            sourceNode.data([sourceData]);
                            targetData.config.up_link.push(sourceID);
                            targetNode.data([targetData]);
                            terra_toolbar.btn.save.disabled = false;
                            $(terra_toolbar.btn.save).children("span").trigger("click");
                        }
                    } else {
                        this.activeNode(node);
                    }
                }
                _onDrag = false;
            };

            let _onmousedown = (event) => {
                console.log(event);
                // svg.bind("mousemove", _onmousemove);
                // _sourceNode = event.target.parentNode;
                // _targetNode = undefined;
            };

            let _onmouseup = (event) => {
                console.log(event);
                // svg.unbind("mousemove", _onmousemove);
                // _targetNode = event.target.parentNode;
                // if(_onContextDrag){
                //     _change_line(true);
                //      this.find(".canvas > .hint").remove();
                // }else if (event.button === 2) {
                //     let params = _cnodes.select(`#${event.currentTarget.id}`).data()[0].config.params;
                //     if (params == null) return;
                //     if (!Object.keys(params).length) return;
                //     let hint = $(`<div class="hint"></div>`),
                //         text = [];
                //     for (let param in params) {
                //         text.push(`${param}: ${params[param].default || ""}`);
                //     }
                //     hint.html(`${text.join("<br />")}`);
                //     hint.css({
                //         left:event.offsetX,
                //         top:event.offsetY,
                //     });
                //     $(".canvas").append(hint);
                // }
                // _onContextDrag = false;
            };

            let _onmousemove = (event) => {
                // if(_onContextDrag){
                //      d3.select("#line-" + _lastLineId)
                //     .attr("x2", event.offsetX)
                //     .attr("y2", event.offsetY);
                // } else{
                //     _create_line();
                //     _onContextDrag = true;
                // }
            };

            // $(document).bind("keydown", (event) => {
            //     if(event.which == 46 || event.which == 8){
            //         if($(".node:hover").length != 0){
            //             _delete_node($(".node:hover")[0]);
            //         } else if($(".line:hover").length != 0){
            //             _delete_line($(".line:hover")[0])
            //         }
            //     }
            // });

            Object.defineProperty(this, "model_schema", {
                set: (schema) => {
                    if(!Array.isArray(schema)) schema = [];
                    _model_schema = schema;
                },
                get: () => {
                    return _model_schema;
                }
            });

            Object.defineProperty(this, "model", {
                set: (model_info) => {
                    _lastNodeId = 0;
                    _lastLineId = 0;
                    _create_model(model_info.layers, model_info.schema);
                    terra_params.reset();
                },
                get: () => {
                    return _cnodes.selectAll("g.node");
                }
            });

            let _lineclick = (event) => {
                _clines.selectAll("line").classed("active", false);
                $(event.currentTarget).addClass("active");
            }

            let _create_line = (dotSource, dotTarget) => {
                let sourceNode = d3.select(dotSource.closest(".node")[0]),
                    sourceID = sourceNode.data()[0].id,
                    sourceMatrix = sourceNode._groups[0][0].transform.baseVal[0].matrix,
                    sourcePosition = [sourceMatrix.e, sourceMatrix.f+_LINE_HEIGHT/2+2];

                let targetNode = dotTarget ? d3.select(dotTarget.closest(".node")[0]) : undefined,
                    targetID = targetNode ? targetNode.data()[0].id : "",
                    targetMatrix = targetNode ? targetNode._groups[0][0].transform.baseVal[0].matrix : undefined,
                    targetPosition = targetMatrix ? [targetMatrix.e, targetMatrix.f-_LINE_HEIGHT/2-2] : undefined;

                let line = _clines.append("line")
                    .attr("id", `line_${sourceID}_${targetID}`)
                    .attr("class", "line")
                    .attr("x1", sourcePosition[0])
                    .attr("y1", sourcePosition[1]);

                if (targetPosition) {
                    line.attr("x2", targetPosition[0])
                        .attr("y2", targetPosition[1]);
                    dotTarget.attr("visibility", "visible");
                } else {
                    _new_link = line;
                    line.attr("x2", sourcePosition[0])
                        .attr("y2", sourcePosition[1]);
                    _new_link._groups[0][0].sourceID = sourceID;
                }

                $(line._groups[0][0]).bind("click", _lineclick);

                dotSource.attr("visibility", "visible");
            }

            // let _create_line = () => {
            //     _lastLineId++;
            //     let _source_node_point = {x:_sourceNode.transform.baseVal[0].matrix.e, y: _sourceNode.transform.baseVal[0].matrix.f};
            //     let line_id =  "line-" + _lastLineId;
            //
            //     let line = _clines.append("line")
            //         .attr("id", line_id)
            //         .attr("class", "line")
            //         .attr("x1", _source_node_point.x + _sourceNode.children[0].width.baseVal.value/2)
            //         .attr("y1", _source_node_point.y + _LINE_HEIGHT)
            //         .attr("x2", _source_node_point.x + _sourceNode.children[0].width.baseVal.value/2)
            //         .attr("y2", _source_node_point.y + _LINE_HEIGHT);
            //
            //     let node_data = _cnodes.select("#" + _sourceNode.id).data()[0];
            //
            //     node_data.lineSource[line_id] = line;
            //     _cnodes.select("#" + _sourceNode.id).select(".dot-source").attr("visibility", "visible");
            //     _cnodes.select("#" + _sourceNode.id).data(node_data);
            // };

            let _update_dots_per_node = (node) => {
                let data = node.data()[0];
                if (data.down_link === undefined) data.down_link = [];
                let clear_links = (items) => {
                    return items.filter((item) => {return item > 0});
                };
                data.down_link = clear_links(data.down_link);
                data.config.up_link = clear_links(data.config.up_link);
                node.select(".dot-source").attr("visibility", data.down_link.length ? "visible" : "hidden");
                node.select(".dot-target").attr("visibility", data.config.up_link.length ? "visible" : "hidden");
            }

            let _clear_links = (node) => {
                let data = node.data()[0];
                if (data.down_link === undefined) data.down_link = [];
                data.down_link.forEach((id) => {
                    _remove_line($(`#line_${data.id}_${id}`));
                });
                data.config.up_link.forEach((id) => {
                    _remove_line($(`#line_${id}_${data.id}`));
                });
            }

            let _create_node = (layer) => {
                layer.lineTarget = {};
                layer.lineSource = {};

                let w = _d3graph._groups[0][0].width.baseVal.value,
                    h = _d3graph._groups[0][0].height.baseVal.value;

                let node = _cnodes.append("g")
                    .attr("id", `node-${layer.id}`)
                    .attr("class", `node node-type-${layer.type}`);

                node.append("circle")
                    .attr("class", "dot dot-target")
                    .attr("visibility", "hidden")
                    .attr("cx", 0)
                    .attr("cy", -_NODE_HEIGHT/2-4);

                node.append("circle")
                    .attr("class", "dot dot-source")
                    .attr("visibility", "hidden")
                    .attr("cx", 0)
                    .attr("cy", _NODE_HEIGHT/2+4);

                let tools = node.append("g").attr("class", "tools"),
                    tools_rect = tools.append("rect").attr("class", "bg").attr("height", 24).attr("y", _NODE_HEIGHT/2-5),
                    rect = node.append("rect").attr("height", _NODE_HEIGHT),
                    text = node.append("text").text(`${layer.config.name}: ${layer.config.type}`),
                    text_box = text._groups[0][0].getBBox(),
                    width = text_box.width + 20;

                let remove = tools.append("rect").attr("class", "btn remove").attr("width", 12).attr("height", 12).attr("y", _NODE_HEIGHT/2+4),
                    link = tools.append("rect").attr("class", "btn link").attr("width", 12).attr("height", 12).attr("y", _NODE_HEIGHT/2+4),
                    unlink = tools.append("rect").attr("class", "btn unlink").attr("width", 12).attr("height", 12).attr("y", _NODE_HEIGHT/2+4);

                let rect_pointer = node.append("rect").attr("class", "pointer").attr("height", _NODE_HEIGHT).call(d3.drag()
                        .on("start", _node_dragstarted)
                        .on("drag", _node_dragged)
                        .on("end", _node_dragended)
                    );

                text.attr("x", -text_box.width/2).attr("y", 12-text_box.height/2);
                rect.attr("width", width).attr("x", -(text_box.width+20)/2).attr("y", -rect._groups[0][0].height.baseVal.value/2);
                rect_pointer.attr("width", width).attr("x", -(text_box.width+20)/2).attr("y", -rect._groups[0][0].height.baseVal.value/2);
                tools_rect.attr("width", width).attr("x", -(text_box.width+20)/2);
                remove.attr("x", (text_box.width+20)/2-16);
                link.attr("x", -(text_box.width+20)/2+4);
                unlink.attr("x", -(text_box.width+20)/2+20);

                $(remove._groups[0][0]).bind("click", (event) => {
                    let g = $(event.currentTarget).closest(".node"),
                        attr_id = g[0].id,
                        node = _d3graph.select(`#${attr_id}`),
                        info = node.data()[0];
                    _clear_links(node);
                    g.remove();
                    if (`${info.id}` === `${$("#field_form-layer_id").val()}`) terra_params.reset();
                    let layers = window.TerraProject.layers;
                    delete layers[info.id];
                    window.TerraProject.layers = layers;
                    terra_toolbar.btn.save.disabled = false;
                    $(terra_toolbar.btn.save).children("span").trigger("click");
                });
                $(link._groups[0][0]).bind("click", (event) => {
                    this.addClass("onlink");
                    _create_line($(event.currentTarget).closest(".node").children(".dot-source"));
                });
                $(unlink._groups[0][0]).bind("click", (event) => {
                    let node = _d3graph.select(`#${$(event.currentTarget).closest(".node")[0].id}`);
                    _clear_links(node);
                    terra_toolbar.btn.save.disabled = false;
                    $(terra_toolbar.btn.save).children("span").trigger("click");
                });

                if (["input", "output"].indexOf(layer.type) > -1) remove.remove();

                if (layer.x === undefined) layer.x = w/2;
                if (layer.y === undefined) layer.y = h/2;

                node.data([layer])
                    .attr("transform", "translate(" + layer.x + "," + layer.y + ")");

                $(node._groups[0][0]).bind("mouseenter", (event) => {
                    _onNode = $(event.currentTarget);
                }).bind("mouseleave", (event) => {
                    if (_new_link) {
                        let node = _d3graph.select(`#${event.currentTarget.id}`),
                            data = node.data()[0],
                            sourceID = _new_link._groups[0][0].sourceID;
                        if (`${sourceID}` !== `${data.id}` && data.config.up_link.indexOf(sourceID) === -1) {
                            _update_dots_per_node(node);
                        }
                    }
                    _onNode = undefined;
                });

                terra_toolbar.btn.save.disabled = false;
                terra_toolbar.btn.clear.disabled = false;

            };

            let _create_model = (layers, schema) => {
                let w = _d3graph._groups[0][0].width.baseVal.value,
                    h = _d3graph._groups[0][0].height.baseVal.value;

                _clines.selectAll("line").remove();
                _cnodes.selectAll("g").remove();
                _lastNodeId = 0;
                _lastLineId = 0;
                _layer_row_w = [];

                let _update_position_by_schema = () => {
                    let rows = schema.length,
                        columns = 0,
                        schema_width = [],
                        columns_width = [],
                        columns_start = [],
                        total_width = 0,
                        start_x = 0,
                        start_y = (h - _NODE_HEIGHT*rows - _LINE_HEIGHT*(rows-1))/2;
                    schema.forEach((indexes, row) => {
                        let length = indexes.length,
                            width = 0;
                        if (length > columns) columns = length;
                        schema_width[row] = [];
                        indexes.forEach((index, column) => {
                            let g = _cnodes.select(`#node-${index}`)._groups[0][0];
                            if (g) {
                                schema_width[row][column] = g.getBBox().width;
                                width += schema_width[row][column];
                            }
                        });
                        if (width > total_width) total_width = width;
                    });
                    total_width += _LINE_HEIGHT*(columns-1);
                    start_x = (w - total_width)/2;
                    for (let i=0; i<columns; i++) {
                        let max_width = 0;
                        for (let k=0; k<rows; k++) {
                            let w = schema_width[k][i] || 0;
                            if (w > max_width) max_width = w;
                        }
                        columns_width[i] = max_width;
                    }
                    columns_width.forEach((width, index) => {
                        columns_start[index] = start_x;
                        if (index > 0) columns_start[index] += columns_width.slice(0, index).reduce((accumulator, currentValue) => accumulator + currentValue) + _LINE_HEIGHT*index;
                        columns_start[index] += width/2;
                    });
                    schema.forEach((indexes, row) => {
                        indexes.forEach((index, column) => {
                            let node = _cnodes.select(`#node-${index}`),
                                g = node._groups[0][0];
                            if (g) {
                                let data = node.data()[0];
                                data.x = columns_start[column];
                                data.y = start_y + _NODE_HEIGHT*(row+1) + row*_LINE_HEIGHT - _LINE_HEIGHT/2;
                                node.data([data]).attr("transform", `translate(${data.x},${data.y})`);
                            }
                        });
                    });
                }

                for (let index in layers) {
                    for (let param in layers[index].config.params) {
                        if (layers[index].config.params[param].type === "tuple") {
                            layers[index].config.params[param].default = `${layers[index].config.params[param].default || ""}`;
                        }
                    }
                }

                let use_schema = false;
                for (let index in layers) {
                    let layer = layers[index];
                    if (layer.x === undefined || layer.y === undefined) use_schema = true;
                    _create_node(layer);
                }
                if (use_schema && schema.length) _update_position_by_schema();

                for (let index in layers) {
                    let layer = layers[index];
                    layer.config.up_link.forEach((item) => {
                        if (item !== 0) {
                            _create_line($(`#node-${item} > .dot-source`), $(`#node-${layer.id} > .dot-target`));
                        }
                    });
                }

                window.TerraProject.layers = layers;
                _d3graph.transition().duration(450).call(zoom.transform, d3.zoomIdentity);
                $(terra_toolbar.btn.save).children("span").trigger("click");
            }

            this.clear = () => {
                window.StatusBar.clear();
                window.ExchangeRequest(
                    "clear_model",
                    (success, data) => {
                        if (success) {
                            this.model = data.data;
                            terra_toolbar.btn.save.disabled = true;
                            terra_toolbar.btn.clear.disabled = true;
                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    },
                )
            }

            $(window).bind("keyup", (event) => {
                if (event.keyCode === 27) {
                    let id = $("#field_form-layer_id").val();
                    if (`${id}` !== "") {
                        terra_params.reset();
                        _d3graph.select(`#node-${id}`).classed("active", false);
                    }
                }
            });

            return this;

        },


        TerraParams: function() {

            if (!this.length) return this;

            let _layer_id_field = $("#field_form-layer_id"),
            _layer_name_field = $("#field_form-layer_name"),
            _layer_type_field = $("#field_form-layer_type"),
            _layer_params_main = this.find(".params-main"),
            _layer_params_extra = this.find(".params-extra"),
            _action_save = this.find(".actions-form > .item.save > button");

            let _render_params = (config) => {
                this.find("#field_form-layer_data").parent().remove();
                _layer_params_main.addClass("hidden");
                _layer_params_main.children(".inner").html("");
                _layer_params_extra.addClass("hidden");
                _layer_params_extra.children(".inner").html("");
                this.find(".params-item.collapsable").addClass("collapsed");
                let _render_params_config = (group, container, params, data) => {
                    let inner = container.children(".inner");
                    if (!Object.keys(params).length) return;
                    for (let name in params) {
                        let param = $.extend(true, {}, params[name]);
                        param.label = name;
                        if (data[name] !== undefined) param.default = data[name];
                        let widget = window.FormWidget(`${group}_${name}`, param);
                        widget.addClass("field-inline");
                        inner.append(widget);
                    }
                    container.removeClass("hidden");
                }
                let params_config = window.TerraProject.layers_types[config.type];
                _render_params_config("main", _layer_params_main, params_config.main, config.params.main);
                _render_params_config("extra", _layer_params_extra, params_config.extra, config.params.extra);
                if (config.data_available && config.data_available.length) {
                    let widget = window.FormWidget("layer_data", {
                        "label":"Данные слоя",
                        "type":"str",
                        "list":true,
                        "default":config.data_name,
                        "available":config.data_available
                    });
                    this.find(".params-config > .inner").append(widget);
                }
            }

            this.reset = () => {
                _layer_id_field.val("");
                _layer_name_field.val("").attr("disabled", "disabled");
                _layer_type_field.val("").attr("disabled", "disabled").selectmenu("refresh");
                this.find("#field_form-layer_data").parent().remove();
                _action_save.attr("disabled", "disabled");
                _layer_params_main.addClass("hidden");
                _layer_params_main.children(".inner").html("");
                _layer_params_extra.addClass("hidden");
                _layer_params_extra.children(".inner").html("");
                this.find(".params-item.collapsable").addClass("collapsed");
            }

            this.load = (data) => {
                this.reset();
                _layer_id_field.val(data.id);
                _layer_name_field.val(data.config.name).removeAttr("disabled");
                _layer_type_field.val(data.config.type).removeAttr("disabled").selectmenu("refresh");
                _action_save.removeAttr("disabled");
                _render_params(data.config);
            }

            let _prepare_data = (serializeData) => {
                let _form = {},
                    _config = {},
                    _params = {};
                for (let item in serializeData) _form[serializeData[item].name] = serializeData[item].value;
                _config = $.extend(true, {}, terra_board.find(`#node-${_form.layer_id}`)[0].__data__);

                _config.config.name = `${_form.layer_name ? _form.layer_name : _config.config.name}`;
                _config.config.type = `${_form.layer_type ? _form.layer_type : _config.config.type}`;
                if (_config.config.data_name !== undefined) _config.config.data_name = _form.layer_data

                _params = window.TerraProject.layers_types[_config.config.type];
                for (let group in _params) {
                    for (let name in _params[group]) {
                        let value = _form[`${group}_${name}`];
                        switch (_params[group][name].type) {
                            case "bool":
                                value = value !== undefined;
                                break;
                            case "int":
                                if (`${value}` !== "") value = parseInt(value);
                                break;
                        }
                        _config.config.params[group][name] = value
                    }
                }
                return _config;
            };

            _layer_type_field.bind("change", (event) => {
                let _config = $.extend(true, {}, terra_board.find(`#node-${_layer_id_field.val()}`)[0].__data__);
                _render_params({
                    "type":event.currentTarget.value,
                    "data_name":_config.config.data_name,
                    "data_available":_config.config.data_available,
                    "params":{"main":{},"extra":{}},
                });
            }).selectmenu({
                change:(event, ui) => {
                    $(event.target).trigger("change");
                }
            });

            this.bind("submit", (event) => {
                event.preventDefault();
                let form = $(event.currentTarget),
                    serializeData = form.serializeArray(),
                    send_data = _prepare_data(serializeData);
                window.StatusBar.clear();
                window.ExchangeRequest(
                    "save_layer",
                    (success, data) => {
                        if (success) {
                            terra_board.model = {"layers":data.data,"schema":[]};
                            window.StatusBar.message(window.Messages.get("LAYER_SAVED"), true);
                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    },
                    send_data
                );
            });

            this.find(".params-item.collapsable > .params-title").bind("click", (event) => {
                event.preventDefault();
                $(event.currentTarget).parent().toggleClass("collapsed");
            });

            return this;

        }


    });


    $(() => {

        terra_toolbar = $(".project-modeling-toolbar").TerraToolbar();
        terra_board = $(".canvas-container").TerraBoard();
        terra_params = $(".params-container").TerraParams();

        terra_toolbar.items.children("span").bind("click", (event) => {
            event.currentTarget.parentNode.execute((item) => {
                if ($(item.parentNode).hasClass("layers")) terra_board.load_layer(item.dataset.type);
            });
        });

        if (!window.TerraProject.dataset) {
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
            terra_board.model = window.TerraProject.model_info;
        }

        LoadModel.find(".model-save-arch-btn > button").bind("click", (event) => {
            window.StatusBar.clear();
            window.ExchangeRequest(
                "set_model",
                (success, data) => {
                    if (success) {
                        window.TerraProject.layers = data.data.layers;
                        window.TerraProject.schema = data.data.schema;
                        terra_board.model = window.TerraProject.model_info;
                        LoadModel.close();
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
                {
                    "layers": event.currentTarget.ModelData.layers,
                    "schema": event.currentTarget.ModelData.front_model_schema
                }
            )
        });

    });

})(jQuery);

