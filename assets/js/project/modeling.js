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
                    // $("#canvas-save").bind("click", () => {
                    //  let nodes = d3.selectAll("g.node").data(),
                    //      send_data = {};
                    //  for(let node in nodes){
                    //      delete nodes[node].lineSource;
                    //      delete nodes[node].lineTarget;
                    //      send_data[nodes[node].id] = nodes[node];
                    //  }
                    //  window.StatusBar.clear();
                    //  window.ExchangeRequest(
                    //      "set_model",
                    //      (success, data) => {
                    //          if (success) {
                    //              this.btn.save.disabled = true;
                    //              terra_board.model = data.data;
                    //              window.StatusBar.message(window.Messages.get("MODEL_SAVED"), true);
                    //              if (typeof callback === "function") callback(item);
                    //          } else {
                    //              window.StatusBar.message(data.error, false);
                    //          }
                    //          },
                    //      {"layers": send_data, "schema": []}
                    //  );
                    // });
                },
                validation: (item, callback) => {
                    window.StatusBar.clear();
                    window.ExchangeRequest(
                        "get_change_validation",
                        (success, data) => {
                            if (success) {
                                console.log(data);
                                this.btn.validation.disabled = true;
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
                    $(this.btn.save).trigger("click");
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

            const _NODE_HEIGHT = 25,
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
                _sourceNode,
                _targetNode,
                _lastNodeId = 0,
                _lastLineId = 0;

            let _layer_row_w_init = (schema) => {
                for(let i=0; i < schema.length; i++){
                    let sum = 0;
                    for(let j=0; j < schema[i].length; j++){
                        if(schema[i][j]){
                            sum += d3.select("#node-"+schema[i][j]).select("rect")._groups[0][0].width.baseVal.value;
                            sum += 50;
                        }
                    }
                    _layer_row_w.push(sum);
                }
            };

            let _set_position_nodes = (schema) => {
                let w = _d3graph._groups[0][0].width.baseVal.value;
                for(let i=0; i <schema.length; i++){
                    let end_nodes = 0,
                        margin_w = (w - _layer_row_w[i])/2,
                        margin_h = 30;
                    for(let j=0; j < schema[i].length; j++){
                        if(schema[i][j]){
                            let node = d3.select("#node-"+schema[i][j]);
                            let node_data = node.data();
                            let node_x = margin_w + end_nodes;
                            let node_y = margin_h + (_LINE_HEIGHT + 30)*i;
                            end_nodes += node.select("rect")._groups[0][0].width.baseVal.value;
                            end_nodes += 50;

                            node_data[0].x = node_x;
                            node_data[0].y = node_y;
                            node.data(node_data);
                            node.attr("transform", "translate(" + node_x + "," + node_y + ")");
                        }
                    }
                }
            };

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

                let input_cfg = {
                    input_shape: [],
                    output_shape: [],
                    params: {},
                    type: "Input",
                    up_link: [0]
                };

                let middle_cfg = {
                    input_shape: [],
                    output_shape: [],
                    params: {
                        activation: {
                            available: [
                                null,
                                "sigmoid",
                                "softmax",
                                "tanh",
                                "relu",
                                "elu",
                                "selu"
                            ],
                            default: "relu",
                            list: true,
                            type: "str"
                        },
                        filters: {
                            default: 16,
                            type: "int"
                        },
                        kernel_size: {
                            default: 3,
                            type: "tuple"
                        },
                        padding: {
                            available: [
                                "valid",
                                "same"
                            ],
                            default: "same",
                            list: true,
                            type: "str"
                        },
                        strides: {
                            default: "1,1",
                            type: "tuple"
                        }
                    },
                    type: "Conv2D",
                    up_link: []
                };

                let output_cfg = {
                    input_shape: [],
                    output_shape: [],
                    params: {
                        activation: {
                            available: [
                                null,
                                "sigmoid",
                                "softmax",
                                "tanh",
                                "relu",
                                "elu",
                                "selu"
                             ],
                            default: "softmax",
                            list: true,
                            type: "str"
                        },
                        units: {
                            default: 3,
                            type: "int"
                        },
                        use_bias: {
                            default: true,
                            type: "bool"
                        }
                    },
                    type: "Dense",
                    up_link: []
                };

                let layer_cfg = {
                    lineTarget: {},
                    lineSource: {}
                };

                switch (class_name){
                    case "input":
                        layer_cfg.config = input_cfg;
                        layer_cfg.type = "input";
                        break

                    case "middle":
                         layer_cfg.config = middle_cfg;
                         layer_cfg.type = "middle";
                        break

                    case "output":
                         layer_cfg.config = output_cfg;
                         layer_cfg.type = "output";
                        break
                }

                _create_node(layer_cfg);
                this.activeNode(d3.select(`#node-${_lastNodeId}`));
            };

            let _create_node = (layer) => {
                layer.lineTarget = {};
                layer.lineSource = {};

                let w = _d3graph._groups[0][0].width.baseVal.value,
                    h = _d3graph._groups[0][0].height.baseVal.value;

                if(new_node){
                    _lastNodeId++;
                     layer.id = _lastNodeId;
                    if(!layer.config.name) layer.config.name = `l${_lastNodeId}_${layer.config.type}`;
                }

                let node = _cnodes.append("g")
                    .attr("id", `node-${layer.id}`)
                    .attr("class", `node node-type-${layer.type}`)
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

                if (layer.x === undefined ) layer.x = w/2;
                if (layer.y === undefined) layer.y = h/2;

                let target_circle = node.append("circle")
                    .attr("class", "dot-target")
                    .attr("visibility", "hidden")
                    .attr("cx", width/2)
                    .attr("cy", -4);

                let source_circle = node.append("circle")
                    .attr("class", "dot-source")
                    .attr("visibility", "hidden")
                    .attr("cx", width/2)
                    .attr("r", 5)
                    .attr("cy", _LINE_HEIGHT);

                node.data([layer])
                    .attr("transform", "translate(" + layer.x + "," + layer.y + ")");

                $(`#node-${layer.id}`).bind("mousedown", _onmousedown)
                    .bind("mouseup", _onmouseup);

                terra_toolbar.btn.save.disabled = false;
                terra_toolbar.btn.validation.disabled = false;
                terra_toolbar.btn.clear.disabled = false;

            };

            let _delete_node = (node) => {

                let target_line = node.__data__.lineTarget,
                    sourse_line = node.__data__.lineSource;

                for(let line in target_line){
                    _delete_line($("#"+line)[0]);
                }
                for(let line in sourse_line){
                    _delete_line($("#"+line)[0]);
                }

                node.remove();
            };

            let _delete_line = (line) => {
                let sourse_node = line.__data__.source._groups[0][0],
                    target_node = line.__data__.target._groups[0][0];

                target_node.__data__.config.up_link.splice(target_node.__data__.config.up_link.indexOf(sourse_node.__data__.id), 1);

                delete sourse_node.__data__.lineSource[line.id];
                delete target_node.__data__.lineTarget[line.id];

                if(Object.keys(sourse_node.__data__.lineSource).length < 1){
                    _cnodes.select("#" + sourse_node.id).select(".dot-source").attr("visibility", "hidden");
                }
                if(Object.keys(target_node.__data__.lineTarget).length < 1){
                     _cnodes.select("#" + target_node.id).select(".dot-target").attr("visibility", "hidden");
                }

                line.remove();
            };

            let _create_line = () => {
                _lastLineId++;
                let _source_node_point = {x:_sourceNode.transform.baseVal[0].matrix.e, y: _sourceNode.transform.baseVal[0].matrix.f};
                let line_id =  "line-" + _lastLineId;

                let line = _clines.append("line")
                    .attr("id", line_id)
                    .attr("class", "line")
                    .attr("x1", _source_node_point.x + _sourceNode.children[0].width.baseVal.value/2)
                    .attr("y1", _source_node_point.y + _LINE_HEIGHT)
                    .attr("x2", _source_node_point.x + _sourceNode.children[0].width.baseVal.value/2)
                    .attr("y2", _source_node_point.y + _LINE_HEIGHT);

                let node_data = _cnodes.select("#" + _sourceNode.id).data()[0];

                node_data.lineSource[line_id] = line;
                _cnodes.select("#" + _sourceNode.id).select(".dot-source").attr("visibility", "visible");
                _cnodes.select("#" + _sourceNode.id).data(node_data);
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


            this.activeNode = (_node) => {
                _cnodes.selectAll(".node").classed("active", false);
                _node.classed("active", true);
                terra_params.load(_node.data()[0]);
            }


            let _node_dragstarted = (data) => {
                this.find(".canvas > .hint").remove();
                let _node = d3.select(`#node-${data.id}`);
                _node.raise().classed("hover", true);
                 if (!_onDrag) this.activeNode(_node);
                 _onDrag = false;
            }

            let _node_dragged = (data) => {
                _onDrag = true;
                let _node = d3.select(`#node-${data.id}`);
                 _node.attr("transform", () => {
                     data.x = d3.event.x;
                     data.y = d3.event.y;
                     return "translate(" + d3.event.x + "," + d3.event.y + ")";
                 });

                 let _node_data = _node.data()[0],
                     lineTarget = _node_data.lineTarget,
                     lineSource = _node_data.lineSource;

                 if (lineTarget) {
                     for(let i in lineTarget) {
                         let cx =  d3.event.x + _node.select("rect")._groups[0][0].width.baseVal.value / 2,
                             cy = d3.event.y;

                         _clines.select("#"+lineTarget[i]._groups[0][0].id)
                             .attr("x2", cx)
                             .attr("y2", cy - 4);
                     }
                 }
                 if (lineSource) {
                     for(let i in lineSource) {
                         let cx = d3.event.x  + _node.select("rect")._groups[0][0].width.baseVal.value / 2,
                             cy = d3.event.y  + _NODE_HEIGHT;

                         _clines.select("#"+lineSource[i]._groups[0][0].id)
                             .attr("x1", cx)
                             .attr("y1", cy + 4);
                     }
                 }
            }

            let _node_dragended = (data) => {
                let _node = d3.select(`#node-${data.id}`);
                _node.classed("hover", false);
                if (!_onDrag) this.activeNode(_node);
                _onDrag = false;
            };

            this.bind("contextmenu", (event) => {
                return false;
            });

            $(document).bind("mousedown", (event) => {
                this.find(".canvas > .hint").remove();
            });

            let _onmousedown = (event)=>{
                svg.bind("mousemove", _onmousemove);
                _sourceNode = event.target.parentNode;
                _targetNode = undefined;
            };

            let _onmouseup = (event)=>{
                svg.unbind("mousemove", _onmousemove);
                _targetNode = event.target.parentNode;
                if(_onContextDrag){
                    _change_line(true);
                     this.find(".canvas > .hint").remove();
                }else if (event.button === 2) {
                    let params = _cnodes.select(`#${event.currentTarget.id}`).data()[0].config.params;
                    if (params == null) return;
                    if (!Object.keys(params).length) return;
                    let hint = $(`<div class="hint"></div>`),
                        text = [];
                    for (let param in params) {
                        text.push(`${param}: ${params[param].default || ""}`);
                    }
                    hint.html(`${text.join("<br />")}`);
                    hint.css({
                        left:event.offsetX,
                        top:event.offsetY,
                    });
                    $(".canvas").append(hint);
                }
                _onContextDrag = false;
            };

            let _onmousemove = (event)=>{
                if(_onContextDrag){
                     d3.select("#line-" + _lastLineId)
                    .attr("x2", event.offsetX)
                    .attr("y2", event.offsetY);
                } else{
                    _create_line();
                    _onContextDrag = true;
                }
            };

            $(document).bind("keydown", (event) => {
                if(event.which == 46 || event.which == 8){
                    if($(".node:hover").length != 0){
                        _delete_node($(".node:hover")[0]);
                    } else if($(".line:hover").length != 0){
                        _delete_line($(".line:hover")[0])
                    }
                }
            });

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

            let _create_model = (layers, schema, new_model) => {
                _clines.selectAll("line").remove();
                _cnodes.selectAll("g").remove();
                _lastNodeId = 0;
                _lastLineId = 0;
                _layer_row_w = [];

                for(let index in layers){
                    for (let param in layers[index].config.params) {
                        if (layers[index].config.params[param].type === "tuple") {
                            layers[index].config.params[param].default = `${layers[index].config.params[param].default || ""}`;
                        }
                    }
                }

                for (let index in layers) {
                    let layer = layers[index];
                    if(layer.id > _lastNodeId){
                        _lastNodeId = layer.id;
                    }
                    _create_node(layer, new_model);
                }

                if(new_model){
                     _layer_row_w_init(schema);
                    _set_position_nodes(schema);
                }

                for (let index in layers) {
                    let layer = layers[index];
                    _targetNode = $("#node-" + layer.id)[0];
                    layer.config.up_link.forEach((parent_node) => {
                        if (parent_node !== 0) {
                            _sourceNode = $("#node-" + parent_node)[0];
                            _create_line();
                            _change_line();
                        }
                    })
                }
                 window.TerraProject.layers = layers;
                _d3graph.transition().duration(450).call(zoom.transform, d3.zoomIdentity);
            }

            this.clear = () => {
                window.TerraProject.model_clear();
                this.model = window.TerraProject.model_info;
                terra_toolbar.btn.save.disabled = true;
                terra_toolbar.btn.validation.disabled = true;
                terra_toolbar.btn.clear.disabled = true;
            }

            return this;

        },


        TerraParams: function() {

            if (!this.length) return this;

            let _layer_id_field = $("#field_form-layer_id"),
            _layer_name_field = $("#field_form-layer_name"),
            _layer_type_field = $("#field_form-layer_type"),
            _layer_params = this.find(".layer-type-params-container"),
            _action_save = this.find(".actions-form > .item.save > button");

            let node,
                node_data;

            this.reset = () => {
                _layer_id_field.val("");
                _layer_name_field.val("").attr("disabled", "disabled");
                _layer_type_field.val("").attr("disabled", "disabled").selectmenu("refresh");
                _action_save.attr("disabled", "disabled");
                _layer_params.addClass("hidden");
                _layer_params.children(".inner").html("");
            }

            this.load = (data) => {
                this.reset();
                _layer_id_field.val(data.id);
                _layer_name_field.val(data.config.name).removeAttr("disabled");
                _layer_type_field.val(data.config.type).removeAttr("disabled").selectmenu("refresh");
                _action_save.removeAttr("disabled");

                node = d3.select("#node-" + data.id)
                node_data = node.data();

                for (let name in data.config.params) {
                    let widget = window.FormWidget(name, data.config.params[name]);
                    widget.addClass("field-inline");
                    _layer_params.children(".inner").append(widget);
                }
                if (data.config.params && Object.keys(data.config.params).length) {
                    _layer_params.removeClass("hidden");
                }
            }

            this.submit = () => {
                throw window.Messages.get("SUBMIT_PARAMS_METHOD");
            }

            let _change_node_data = (node_data, serializeData) => {
                for (let index in serializeData) {
                    if(node_data[0].config.params != null && node_data[0].config.params[serializeData[index].name]){
                        switch (node_data[0].config.params[serializeData[index].name].type){
                            case "int":
                                serializeData[index].value = parseInt(serializeData[index].value);
                                break

                            case "str":
                                break

                            case "tuple":
                                break

                            case "bool":
                                 serializeData[index].value = serializeData[index].value == 'true';
                                break
                        }
                        node_data[0].config.params[serializeData[index].name].default = serializeData[index].value;
                    } else if(serializeData[index].name == "layer_type"){
                        node_data[0].config.type = serializeData[index].value;
                    } else if(serializeData[index].name == "layer_name"){
                        node_data[0].config.name = serializeData[index].value;
                    }
                }

                return node_data;
            };

            this.bind("submit", (event) => {
                event.preventDefault();
                let form = $(event.currentTarget),
                    serializeData = form.serializeArray();
                _change_node_data(node_data, serializeData);

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
                            terra_board.model = data.data;
                            window.StatusBar.message(window.Messages.get("MODEL_SAVED"), true);
                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    },
                    {"layers": send_data, "schema": []}
                );

            });

            return this;

        }


    });


    $(() => {

        terra_toolbar = $(".project-modeling-toolbar").TerraToolbar();
        terra_board = $(".canvas-container").TerraBoard();
        terra_params = $(".params-container").TerraParams();

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
                        console.log(data.data.layers);
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

        terra_toolbar.items.children("span").bind("click", (event) => {
            event.currentTarget.parentNode.execute((item) => {
                if ($(item.parentNode).hasClass("layers")) terra_board.load_layer(item.dataset.type);
            });
        });

    });

})(jQuery);

