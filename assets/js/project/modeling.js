"use strict";


(($) => {


    let LoadModel = $("#modal-window-load-model").ModalWindow({
        title:window.Messages.get("LOAD_MODEL"),
        width:680,
        height:440,
        request:["get_models"]
    });


    let Toolbar = function(el) {

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
                                        window.StatusBar.message(data.error, true);
                                    }
                                },
                                {"model_file":item.data("name")}
                            );
                        });
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
            el.find(".menu-section.layers > li[data-type=input]")[0].disabled = input;
            el.find(".menu-section.layers > li[data-type=middle]")[0].disabled = middle;
            el.find(".menu-section.layers > li[data-type=output]")[0].disabled = output;
        }

        Object.defineProperty(this, "items", {
            get: () => {
                return el.find(".menu-section > li");
            }
        });

        Object.defineProperty(this, "buttons", {
            get: () => {
                return this.items.children("span");
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

        this.buttons.each((index, button) => {
            button.item = button.parentNode;
        });

    }


    let Params = function(el) {

        let _layer_index_field = $("#field_form-layer_index"),
            _layer_name_field = $("#field_form-layer_name"),
            _layer_type_field = $("#field_form-layer_type"),
            _layer_params = el.find(".layer-type-params-container"),
            _action_save = el.find(".actions-form > .item.save > button");

        this.reset = () => {
            _layer_index_field.val("");
            _layer_name_field.val("").attr("disabled", "disabled");
            _layer_type_field.val("").attr("disabled", "disabled").selectmenu("refresh");
            _action_save.attr("disabled", "disabled");
            _layer_params.addClass("hidden");
            _layer_params.children(".inner").html("");
        }

        this.load = (data) => {
            this.reset();
            _layer_index_field.val(data.index);
            _layer_name_field.val(data.config.name).removeAttr("disabled");
            _layer_type_field.val(data.config.type).removeAttr("disabled").selectmenu("refresh");
            _action_save.removeAttr("disabled");
            for (let name in data.config.params) {
                let widget = window.FormWidget(name, data.config.params[name]);
                widget.addClass("field-inline");
                _layer_params.children(".inner").append(widget);
            }
            if (Object.keys(data.config.params).length) {
                _layer_params.removeClass("hidden");
            }
        }

        this.submit = () => {
            throw window.Messages.get("SUBMIT_PARAMS_METHOD");
        }

        el.bind("submit", (event) => {
            event.preventDefault();
            let form = $(event.currentTarget),
                serializeData = form.serializeArray(),
                data = {};
            for (let index in serializeData) {
                data[serializeData[index].name] = serializeData[index].value;
            }
            this.submit(data);
        });

    }


    let Model = function(el, params, toolbar) {

        const NODE_HEIGHT = 25,
            DEFAULT_LINE_HEIGHT = 30;

        let _d3graph = d3.select(el.find(".canvas > svg")[0]),
            _clines = _d3graph.select("#canvas-lines"),
            _cnodes = _d3graph.select("#canvas-nodes"),
            _onDrag = false,
            _onContextDrag = false,
            _sourceNode,
            _targetNode;

        el.bind("contextmenu", (event) => {
            return false;
        });
        $(document).bind("mousedown", (event) => {
            el.find(".canvas > .hint").remove();
        });

        let _node_dragstarted = (data) => {
            el.find(".canvas > .hint").remove();
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
             let _node_data = _node.data()[0],
                 lineTarget = _node_data.lineTarget,
                 lineSource = _node_data.lineSource;
             if (lineTarget) {
                 for(let i in lineTarget) {
                     let cx = _node_data.x + _node.select("rect")._groups[0][0].width.baseVal.value / 2,
                         cy = _node_data.y,
                         ox = lineTarget[i]._groups[0][0].transform.baseVal[0].matrix.e,
                         oy = lineTarget[i]._groups[0][0].transform.baseVal[0].matrix.f;
                     lineTarget[i].select(".dot-target")
                         .attr("cx", cx - ox - 1)
                         .attr("cy", cy - oy - 4);
                     lineTarget[i].select("line")
                         .attr("x2", cx - ox - 1)
                         .attr("y2", cy - oy - 4);
                 }
             }
             if (lineSource) {
                 for(let i in lineSource) {
                     let cx = _node_data.x + _node.select("rect")._groups[0][0].width.baseVal.value / 2,
                         cy = _node_data.y + NODE_HEIGHT,
                         ox = lineSource[i]._groups[0][0].transform.baseVal[0].matrix.e,
                         oy = lineSource[i]._groups[0][0].transform.baseVal[0].matrix.f;
                     lineSource[i].select(".dot-source")
                         .attr("cx", cx - ox - 1)
                         .attr("cy", cy - oy + 4);
                     lineSource[i].select("line")
                         .attr("x1", cx - ox - 1)
                         .attr("y1", cy - oy + 4);
                 }
             }
        }

        let _node_dragended = (data) => {
            let _node = d3.select(`#node-${data.index}`);
            _node.classed("hover", false);
            if (!_onDrag) this.activeNode(_node);
            _onDrag = false;
        }

        let _create_node = (layer, count) => {

            let w = _d3graph._groups[0][0].width.baseVal.value,
                h = _d3graph._groups[0][0].height.baseVal.value,
                svg = $(_d3graph._groups[0][0]);

            layer.lineTarget = [];
            layer.lineSource = [];

            let node = _cnodes.append("g")
                .attr("id", `node-${layer.index}`)
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

            if (layer.x === undefined) layer.x = (w - width) / 2;
            if (layer.y === undefined) layer.y = (h - (count * NODE_HEIGHT + (count - 1) * DEFAULT_LINE_HEIGHT)) / 2 + (parseInt(layer.index) - 1) * (NODE_HEIGHT + DEFAULT_LINE_HEIGHT);

            node.data([layer])
                .attr("transform", "translate(" + layer.x + "," + layer.y + ")");

            // $(node._groups[0][0]).bind("mouseup", (event) => {
            //     if (event.button === 2 && !_onContextDrag) {
            //         let params = _cnodes.select(`#${event.currentTarget.id}`).data()[0].config.params;
            //         if (!Object.keys(params).length) return;
            //         let hint = $(`<div class="hint"></div>`),
            //             text = [];
            //         for (let param in params) {
            //             text.push(`${param}: ${params[param].default || ""}`);
            //         }
            //         hint.html(`${text.join("<br />")}`);
            //         hint.css({
            //             left:event.offsetX,
            //             top:event.offsetY,
            //         });
            //         el.children(".canvas").append(hint);
            //     }
            //      _onContextDrag = false;
            // });

            let _onmousedown = (event)=>{
                svg.bind("mousemove", _onmousemove);
                _sourceNode = event.target.parentNode;
            };

            let _onmouseup = (event)=>{
                svg.unbind("mousemove", _onmousemove);
                _targetNode = event.target.parentNode;
                _onContextDrag = false;
                _create_custom_line();
            };

            let _onmousemove = (event)=>{
                _onContextDrag = true;
            };

            $(".node").bind("mousedown", _onmousedown)
                .bind("mouseup", _onmouseup);
        }

        let _create_custom_line = () => {

            let _source_node_point = {x:_sourceNode.transform.baseVal[0].matrix.e, y: _sourceNode.transform.baseVal[0].matrix.f};
            let _target_node_point = {x: _targetNode.transform.baseVal[0].matrix.e, y: _targetNode.transform.baseVal[0].matrix.f};

            let line = _clines.append("g")
                .attr("id", `line-${$("line").length + 1}`)
                .attr("class", "line")
                .attr("transform", `translate(${0}, ${0})`);
            line.append("line")
                .attr("x1", _source_node_point.x + _sourceNode.children[0].width.baseVal.value/2)
                .attr("y1", _source_node_point.y + DEFAULT_LINE_HEIGHT)
                .attr("x2", _target_node_point.x + _targetNode.children[0].width.baseVal.value/2)
                .attr("y2", _target_node_point.y);
            line.append("circle")
                .attr("class", "dot-source")
                .attr("cx", _source_node_point.x + _sourceNode.children[0].width.baseVal.value/2)
                .attr("cy", _source_node_point.y + DEFAULT_LINE_HEIGHT)
            line.append("circle")
                .attr("class", "dot-target")
                .attr("cx", _target_node_point.x + _targetNode.children[0].width.baseVal.value/2)
                .attr("cy", _target_node_point.y - 4);
             line.data([{source:  _cnodes.select("#" + _sourceNode.id), target:  _cnodes.select("#" + _targetNode.id)}]);

            let node_data = _cnodes.select("#" + _sourceNode.id).data()[0],
                next_node_data = _cnodes.select("#" + _targetNode.id).data()[0];

            node_data.lineSource.push(line);
            next_node_data.lineTarget.push(line) ;
            _cnodes.select("#" + _sourceNode.id).data(node_data);
            _cnodes.select("#" + _targetNode.id).data(next_node_data);

        }

        let _create_line = (layer) => {
            let _node = this.getNodeByIndex(layer.index),
                _node_data = _node.data()[0],
                _rect = _node.select("rect")._groups[0][0],
                _next_node = this.getNodeByIndex(parseInt(layer.index) + 1);

            if (!_next_node._groups[0][0]) return null;
            let _next_node_data = _next_node.data()[0];
            let line = _clines.append("g")
                .attr("id", `line-${layer.index}`)
                .attr("class", "line")
                .attr("transform", `translate(${_node_data.x + _rect.width.baseVal.value / 2},${_node_data.y + NODE_HEIGHT})`);
            line.append("line")
                .attr("x1", -1)
                .attr("y1", 3)
                .attr("x2", -1)
                .attr("y2", DEFAULT_LINE_HEIGHT - 4);
            line.append("circle")
                .attr("class", "dot-source")
                .attr("cx", -1)
                .attr("cy", 4);
            line.append("circle")
                .attr("class", "dot-target")
                .attr("cx", -1)
                .attr("cy", DEFAULT_LINE_HEIGHT - 4);
            line.data([{source: _node, target: _next_node}]);
            _node_data.lineSource.push(line);
            _node.data(_node_data);
            _next_node_data.lineTarget.push(line);
            _next_node.data(_next_node_data);
        }



        let _create_model = (layers) => {
            layers.forEach((layer) => {
                _create_node(layer, Object.keys(layers).length);
            });
            layers.forEach((layer) => {
                _create_line(layer);
            });
        }

        let _existsLayersTypes = () => {
            let _layers = this.layers_config,
                input = false,
                middle = false,
                output = false;
            for  (let index in _layers) {
                switch (_layers[index].type) {
                    case "input":
                        input = true;
                        break;
                    case "middle":
                        middle = true;
                        break;
                    case "output":
                        output = true;
                        break;
                }
            }
            return [input, middle, output];
        }

        Object.defineProperty(this, "layer", {
            set: (value) => {
                if (!value.config.params) value.config.params = {};
                for (let param in value.config.params) {
                    if (value.config.params[param].type === "tuple") {
                        value.config.params[param].default = `${value.config.params[param].default || ""}`;
                    }
                }
            }
        });

        Object.defineProperty(this, "layers", {
            set: (value) => {
                this.clear();
                let num = 0,
                    _layer,
                    _layers = [];
                for (let index in value) {
                    let type = "middle";
                    if (num === Object.keys(value).length - 1) type = "output";
                    if (num === 0) type = "input";
                    _layer = {
                        index:index,
                        config:value[index],
                        type:type
                    };
                    _layers.push(_layer);
                    this.layer = _layer
                    num++;
                }
                _create_model(_layers);
                let exists = _existsLayersTypes();
                toolbar.layersReset(exists[0], exists[1], exists[2]);
                params.reset();
            },
            get: () => {
                return _cnodes.selectAll("g.node");
            }
        });

        Object.defineProperty(this, "layers_config", {
            get: () => {
                let _layers = this.layers.data(),
                    output = {};
                for (let i=0; i<_layers.length; i++) {
                    output[_layers[i].index] = {
                        type:_layers[i].type,
                        config:_layers[i].config
                    };
                }
                return output;
            }
        });

        this.activeNode = (_node) => {
            _cnodes.selectAll(".node").classed("active", false);
            _node.classed("active", true);
            params.load(_node.data()[0]);
        }

        this.getNodeByIndex = (index) => {
            return _cnodes.select(`#node-${index}`);
        }

        this.clear = () => {
            _clines.selectAll("g").remove();
            _cnodes.selectAll("g").remove();
        }

        this.layers = {}

    }


    let ModelEditor = function(options) {

        let _toolbar = new Toolbar(options.toolbar),
            _params = new Params(options.params),
            _model = new Model(options.board, _params, _toolbar);

        Object.defineProperty(this, "model", {
            set: (value) => {
                _model.layers = value;
            },
            get: () => {
                return _model;
            }
        });

        _params.submit = (send_data) => {
            let data = {},
                layers = $.extend(true, {}, _model.layers_config);
            for (let index in layers) {
                let config = layers[index].config;
                if (index === send_data.layer_index) {
                    config.name = send_data.layer_name;
                    config.type = send_data.layer_type;
                    for (let param in config.params) {
                        config.params[param] = send_data[param] ? `${send_data[param]}` : "";
                    }
                } else {
                    for (let param in config.params) {
                        config.params[param] = config.params[param].default ? `${config.params[param].default}` : "";
                    }
                }
                data[index] = config;
            }
            _params.reset();
            window.StatusBar.clear();
            window.ExchangeRequest(
                "get_change_validation",
                (success, data) => {
                    if (success) {
                        _model.layers = data.data.layers;
                    } else {
                        window.StatusBar(data.error, false);
                    }
                },
                {layers:data}
            );
        }

        let _loadLayer = (layer) => {
            window.StatusBar.message(window.Messages.get("TRYING_TO_LOAD_LAYER", [layer]));
            let _configs = _model.layers_config,
                _input_exists = false,
                _output_exists = false,
                _method = `set_${layer === "input" ? "input" : "any"}_layer`,
                _condition_layer_type_exists;
            for (let i in _configs) {
                if (_configs[i].type === "input") _input_exists = true;
                if (_configs[i].type === "output") _output_exists = true;
            }
            if (["input", "output"].indexOf(layer) > -1) {
                _condition_layer_type_exists = layer === "input" ? _input_exists : _output_exists;
            }
            if (_condition_layer_type_exists === true) {
                window.StatusBar.message(window.Messages.get("LAYER_ALREADY_EXISTS", [layer]), false);
            } else {
                let send_data = {};
                if (layer === "output") send_data = {layer_type:"output"}
                window.ExchangeRequest(
                    _method,
                    (success, data) => {
                        if (success) {
                            window.StatusBar.message(window.Messages.get("LAYER_LOADED", [layer]), true);
                            let _layers = data.data.layers,
                                _indexes = Object.keys(_layers),
                                _input_layer = _indexes.length ? _indexes[0] : undefined,
                                _middle_layer = _indexes.length > 2 ? _indexes[_indexes.length - 2] : undefined,
                                _output_layer = _indexes.length > 1 ? _indexes[_indexes.length - 1] : undefined,
                                _load_index;
                            switch (layer) {
                                case "input":
                                    _load_index = _input_layer;
                                    break;
                                case "middle":
                                    _load_index = _middle_layer;
                                    break;
                                case "output":
                                    _load_index = _output_layer;
                                    break;
                            }
                            _model.layers = data.data.layers;
                            if (_load_index !== undefined) {
                                _model.activeNode(_model.getNodeByIndex(_load_index));
                            }
                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    },
                    send_data
                );
            }
        }

        _toolbar.buttons.bind("click", (event) => {
            event.preventDefault();
            event.currentTarget.item.execute((item) => {
                if ($(item.parentNode).hasClass("layers")) _loadLayer(item.dataset.type);
            });
        });

    }


    $(() => {

        let me = new ModelEditor({
            toolbar:$(".project-modeling-toolbar"),
            board:$(".canvas-container"),
            params:$(".params-container")
        });

        if (!window.TerraProject.dataset || !window.TerraProject.task) {
            let warning = $("#modal-window-warning").ModalWindow({
                title:"Предупреждение!",
                width:300,
                height:174,
                noclose:true,
                callback:(data) => {
                    warning.children(".wrapper").append($(`
                        <p>Для редактирования модели необходимо загрузить датасет.</p>
                        <p><a class="format-link" href="/project/datasets/">Загрузить датасет</a></p>
                    `));
                }
            });
            warning.open();
        } else {
            me.model = window.TerraProject.layers;
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

    })

    addEventListener("keydown", (event)=>{
        if(event.which == 46){
            if($(".node:hover").length != 0){

                for(let i in $(".node:hover")[0].__data__.lineTarget){
                    $(".node:hover")[0].__data__.lineTarget[i]._groups[0][0].remove();
                }
                for(let i in $(".node:hover")[0].__data__.lineSource){
                    $(".node:hover")[0].__data__.lineSource[i]._groups[0][0].remove();
                }
                $(".node:hover").remove();
            } else if($(".line:hover").length != 0){
                $(".line:hover").remove();
            }
        }
    });

})(jQuery);
