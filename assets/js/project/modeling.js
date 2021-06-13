"use strict";


(($) => {


    let terra_toolbar, terra_board, terra_params;
    

    let LoadModel = $("#modal-window-load-model").ModalWindow({
        title:window.Messages.get("LOAD_MODEL"),
        width:680,
        height:440,
        request:["get_models"]
    });


    let KerasCode = $("#modal-window-keras-code").ModalWindow({
        title:"Код на Keras",
        width:680,
        height:440,
        request:["get_keras_code"],
        callback:(ui, data) => {
            ui.find(".action > .result").text("");
            let map_replace = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&#34;',
                "'": '&#39;'
            };
            ui.find(".wrapper .content").html(`<pre>${data.code.replace(/[&<>'"]/g, (c) => {return map_replace[c]})}</pre>`);
        }
    });


    let SaveModel = $("#modal-window-save-model").ModalWindow({
        title:"Сохранение модели",
        width:400,
        height:212
    });


    let ClearModel = $("#modal-window-clear-model").ModalWindow({
        title:"Очистить модель",
        width:300,
        height:160
    });


    let fallbackCopyTextToClipboard = (text) => {
        let textArea = document.createElement("textarea"),
            success = false;
        textArea.value = text;
        textArea.style.top = "0";
        textArea.style.left = "0";
        textArea.style.position = "fixed";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
            success = document.execCommand('copy');
        } catch (err) {
            success = false;
        }
        document.body.removeChild(textArea);
        return success;
    }


    let clip = (el) => {
        let range = document.createRange();
        range.selectNodeContents(el);
        let sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
    }


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
                                block.find(".models-data > .models-list .loaded-list").append($(`<li data-name="${data[index].name}" data-is_terra="${data[index].is_terra}"><span>${data[index].name}</span></li>`))
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
                                            block.find(".models-data > .model-arch > .wrapper > .model-arch-img > img").attr("src", `data:image/png;base64,${data.data.preview.preview_image}`);
                                            block.find(".models-data > .model-arch > .wrapper").removeClass("hidden");
                                            block.find(".models-data > .model-arch > .wrapper > .model-save-arch-btn > button")[0].ModelData = data.data;
                                            block.find(".models-data > .model-arch > .wrapper > .modal-arch-info > .name > span").text(data.data.preview.name);
                                            block.find(".models-data > .model-arch > .wrapper > .modal-arch-info > .datatype > span").text(data.data.preview.datatype);
                                            let input_shape = [];
                                            for (let output_name in data.data.preview.input_shape) {
                                                input_shape.push(JSON.stringify(data.data.preview.input_shape[output_name]));
                                            }
                                            block.find(".models-data > .model-arch > .wrapper > .modal-arch-info > .input_shape > span").text(input_shape.join(", "));
                                            terra_toolbar.btn.keras.disabled = false;
                                        } else {
                                            window.StatusBar.message(data.error, false);
                                        }
                                    },
                                    {"model_file":item.data("name"),"is_terra":item.data("is_terra")}
                                );
                            });
                        }
                    );
                },
                save_model: (item, callback) => {
                    window.StatusBar.clear();
                    SaveModel.open((target) => {
                        $("#field_form-save_model_name").val(window.TerraProject.model_name).focus();
                    });
                },
                save: (item, callback) => {
                    let send_data = {};
                    d3.selectAll("g.node")._groups[0].forEach((item) => {
                        send_data[parseInt(item.dataset.index)] = item.__data__;
                    });
                    window.StatusBar.clear();
                    window.ExchangeRequest(
                        "set_model",
                        (success, data) => {
                            if (success) {
                                this.btn.save.disabled = true;
                                this.btn.save_model.disabled = true;
                                this.btn.keras.disabled = true;
                                if (typeof callback === "function") callback(item);
                            } else {
                                window.StatusBar.message(data.error, false);
                            }
                        },
                        {
                            "layers": send_data,
                            "schema": window.TerraProject.schema,
                        }
                    );
                },
                validation: (item, callback) => {
                    this.btn.validation.disabled = true;
                    window.StatusBar.clear();
                    window.StatusBar.message(window.Messages.get("VALIDATE_MODEL"));
                    terra_board.model.classed("error", false);
                    terra_toolbar.btn.save_model.disabled = true;
                    terra_toolbar.btn.keras.disabled = true;
                    terra_board.model.selectAll("g.errors").remove();
                    window.ExchangeRequest(
                        "get_change_validation",
                        (success, data) => {
                            this.btn.validation.disabled = false;
                            window.TerraProject.logging = data.data.logging;
                            if (success) {
                                window.StatusBar.clear();
                                for (let index in data.data.errors) {
                                    let error = data.data.errors[index];
                                    terra_board.set_layer_error(index, error);
                                    if (error) window.StatusBar.message(window.Messages.get("VALIDATION_MODEL_ERROR"), false);
                                }
                                if (data.data.validated) {
                                    terra_toolbar.btn.save_model.disabled = false;
                                    terra_toolbar.btn.keras.disabled = false;
                                    window.StatusBar.message(window.Messages.get("VALIDATION_MODEL_SUCCESS"), true);
                                }
                            } else {
                                window.StatusBar.message(data.error, false);
                            }
                        }
                    );
                },
                keras: (item, callback) => {
                    window.StatusBar.clear();
                    KerasCode.open();
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
                        "save_model":this.find(".menu-section > li[data-type=save_model]")[0],
                        "save":this.find(".menu-section > li[data-type=save]")[0],
                        "validation":this.find(".menu-section > li[data-type=validation]")[0],
                        "keras":this.find(".menu-section > li[data-type=keras]")[0],
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

            const _NODE_HEIGHT = 40,
                _LINE_HEIGHT = 30,
                _SCALE_FACTOR = .2,
                _TRANSITION_DURATION = 300;

            let _d3graph = d3.select(this.find(".canvas > svg")[0]),
                _cc = _d3graph.select("#canvas-container"),
                _clines = _d3graph.select("#canvas-lines"),
                _cnodes = _d3graph.select("#canvas-nodes"),
                _zoom = this.find("ul.zoom"),
                svg = $(_d3graph._groups[0][0]),
                zoom = d3.zoom().scaleExtent([.5, 2]).on("zoom", () => {_cc.attr("transform", d3.event.transform)}),
                _layer_row_w = [],
                _model_schema = [],
                _onDrag = false,
                _onNode,
                _new_link;

            let _zoom_call = {
                inc: () => {
                    zoom.scaleBy(_d3graph.transition().duration(_TRANSITION_DURATION), 1 + _SCALE_FACTOR);
                },
                dec: () => {
                    zoom.scaleBy(_d3graph.transition().duration(_TRANSITION_DURATION), 1 - _SCALE_FACTOR);
                },
                reset: () => {
                    _d3graph.transition().duration(_TRANSITION_DURATION).call(zoom.transform, d3.zoomIdentity);
                }
            }

            _zoom.find("li > span").bind("click", (event) => {
                event.preventDefault();
                let method = $(event.currentTarget).parent().data("type");
                if (typeof _zoom_call[method] === "function") _zoom_call[method]();
            });

            _d3graph.call(zoom);

            let _separate_to_multiline = (text, max_length=20) => {
                if (!text) return [];
                let output = [],
                    words = text.split(/\s/);
                if (words.length) {
                    while (words.length) {
                        let line = "";
                        while (line.length < max_length && words.length) {
                            line += `${words.shift()} `;
                        }
                        output.push(line);
                    }
                }
                return output;
            }

            this.set_layer_error = (index, message) => {
                let _node = _cnodes.select(`#node-${index}`),
                    _errors_list = _separate_to_multiline(message || "");
                _node.selectAll("g.errors").remove();
                if (!_errors_list.length) return;

                let _node_size = _node.select("rect.node-bg")._groups[0][0].getBBox();
                _node.classed("error", true);
                let _error_group = _node.append("g").attr("class", "errors"),
                    _error_rect = _error_group.append("rect").attr("class", "error-bg").attr("height", _NODE_HEIGHT),
                    _error_text = _error_group.append("text");
                if (_errors_list.length) {
                    _errors_list.forEach((item, index) => {
                        _error_text.append("tspan").text(item).attr("x", 10).attr("y", index*16+18-_NODE_HEIGHT/2+2).attr("fill", "#ffffff").attr("font-size", "12px");
                    });
                }
                let _errors_box = _error_text._groups[0][0].getBBox();
                _error_rect.attr("width", _errors_box.width+20).attr("height", _errors_box.height+14).attr("x", -(_errors_box.width+24)-_node_size.width/2).attr("y", -_NODE_HEIGHT/2);
                _error_text.selectAll("tspan").attr("x", -(_errors_box.width+14)-_node_size.width/2);
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

                terra_toolbar.btn.middle.disabled = true;
                let indexes = Object.keys(window.TerraProject.layers).map((value) => {
                    return parseInt(value);
                });
                if (!indexes.length) indexes = [0];
                let _max_id = Math.max.apply(Math, indexes)+1;

                let layer_config = {
                    name: `L${_max_id}`,
                    type: type,
                    location_type: class_name,
                    params: $.extend(true, {}, window.TerraProject.layers_types[type]),
                };

                for (let group in layer_config.params) {
                    for (let param in layer_config.params[group]) {
                        layer_config.params[group][param] = layer_config.params[group][param].default;
                    }
                }

                let layer_default = {
                    config: layer_config,
                    down_link: [],
                    x: null,
                    y: null,
                };

                terra_toolbar.btn.save_model.disabled = true;
                terra_toolbar.btn.keras.disabled = true;
                window.ExchangeRequest(
                    "save_layer",
                    (success, data) => {
                        terra_toolbar.btn.middle.disabled = false;
                        if (success) {
                            _create_node(parseInt(data.data.index), data.data.layers[parseInt(data.data.index)]);
                            window.TerraProject.layers = data.data.layers;
                            this.activeNode(d3.select(`#node-${data.data.index}`)._groups[0][0]);
                        }
                    },
                    {
                        "index": _max_id,
                        "layer": layer_default
                    }
                );
            };

            this.activeNode = (g, no_reload) => {
                let node = _cnodes.select(`#${g.id}`);
                _cnodes.selectAll(".node").classed("active", false);
                node.classed("active", true);
                if (!no_reload) terra_params.load(parseInt(g.dataset.index), node.data()[0]);
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
                    let id = $("#field_form-index").val();
                    if (`${id}` !== "") {
                        terra_params.reset();
                        _d3graph.select(`#node-${id}`).classed("active", false);
                    }
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
                    if (_onNode && `${_new_link._groups[0][0].sourceID}` !== `${_onNode[0].dataset.index}` && _onNode[0].__data__.config.up_link.indexOf(_new_link._groups[0][0].sourceID) === -1) {
                        let matrix = _onNode[0].transform.baseVal[0].matrix;
                        x2 = matrix.e;
                        y2 = matrix.f - _LINE_HEIGHT / 2 - 9;
                        _onNode.children(".dot-target").attr("visibility", "visible");
                    } else {
                        let transform = _get_transform();
                        x2 = (x2 - transform.x)/transform.s;
                        y2 = (y2 - transform.y)/transform.s;
                    }
                    _new_link.attr("x2", x2).attr("y2", y2);
                }
            });

            let _get_transform = () => {
                let baseVal = _cc._groups[0][0].transform.baseVal,
                    output = {x:0,y:0,s:1};
                if (baseVal.length) {
                    for (let i=0; i<baseVal.length; i++) {
                        switch (baseVal[i].type) {
                            case 2:
                                output.x = baseVal[i].matrix.e;
                                output.y = baseVal[i].matrix.f;
                                break;
                            case 3:
                                output.s = baseVal[i].matrix.a;
                                break;
                        }
                    }
                }
                return output;
            }

            let _node_dragged = (data, _, rect) => {
                let node = $(rect).parent()[0],
                    _node = d3.select(`#${node.id}`),
                    info = _node.data()[0];
                _onDrag = true;
                _node.attr("transform", (data) => {
                    let transform = _get_transform();
                    data.x = (d3.event.sourceEvent.layerX - transform.x)/transform.s - d3.event.subject.x;
                    data.y = (d3.event.sourceEvent.layerY - transform.y)/transform.s - d3.event.subject.y;
                    return `translate(${data.x},${data.y})`;
                }).raise();

                let matrix = node.transform.baseVal[0].matrix,
                    x = matrix.e, y = matrix.f;

                info.down_link.forEach((index) => {
                    _clines.select(`#line_${node.dataset.index}_${index}`).attr("x1", x).attr("y1", y+_LINE_HEIGHT/2+9);
                });
                info.config.up_link.forEach((index) => {
                    _clines.select(`#line_${index}_${node.dataset.index}`).attr("x2", x).attr("y2", y-_LINE_HEIGHT/2-9);
                });
            }

            let _node_dragended = (data, _, rect) => {
                let node = $(rect).parent()[0],
                    _node = d3.select(`#${node.id}`);
                if (_onDrag) {
                    terra_toolbar.btn.save_model.disabled = true;
                    terra_toolbar.btn.keras.disabled = true;
                    window.ExchangeRequest(
                        "save_layer",
                        null,
                        {
                            "index": parseInt(node.dataset.index),
                            "layer": _node.data()[0]
                        }
                    );
                } else {
                    if (this.hasClass("onlink")) {
                        let matrix = node.transform.baseVal[0].matrix,
                            sourceID = parseInt(_new_link._groups[0][0].sourceID),
                            sourceNode = _d3graph.select(`#node-${sourceID}`),
                            sourceData = sourceNode.data()[0],
                            targetID = parseInt(node.dataset.index),
                            targetNode = _d3graph.select(`#node-${targetID}`),
                            targetData = targetNode.data()[0];
                        if (`${sourceID}` !== `${targetID}` && targetData.config.up_link.indexOf(sourceID) === -1) {
                            this.removeClass("onlink");
                            _new_link.attr("id", `line_${sourceID}_${targetID}`)
                                .attr("x2", matrix.e)
                                .attr("y2", matrix.f - _LINE_HEIGHT / 2 - 9);
                            _new_link = undefined;
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
                    _create_model(model_info.layers, model_info.schema);
                    if (!model_info.no_close) terra_params.reset();
                },
                get: () => {
                    return _cnodes.selectAll("g.node");
                }
            });

            Object.defineProperty(this, "svg", {
                get: () => {
                    let _d3_svg = d3.select(svg.clone()[0]);
                    _d3_svg.attr("width", svg.width())
                        .attr("height", svg.height());
                    _d3_svg.selectAll("g.tools").remove();
                    _d3_svg.selectAll("g.params").remove();
                    _d3_svg.selectAll("rect.pointer").remove();
                    _d3_svg.selectAll("tspan.name");
                    _d3_svg.selectAll("tspan.params");
                    let _svg_o = $(_d3_svg._groups[0][0]);
                    _svg_o.find("circle[visibility=hidden]").remove();
                    _svg_o.find("g, line, rect, circle").removeAttr("id").removeAttr("class").removeAttr("data-index").removeAttr("visibility");
                    _svg_o.prepend(`<defs><style type="text/css">@font-face{font-family:"Open Sans";src:url('data:font/woff2;base64,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')}</style></defs>`);
                    let _svg = _svg_o[0];
                    _svg.setAttribute("xlink", "http://www.w3.org/1999/xlink");
                    let _serializer = new XMLSerializer(),
                        _svg_string = _serializer.serializeToString(_svg);
                    _svg_string = _svg_string.replace(/(\w+)?:?xlink=/g, 'xmlns:xlink='); // Fix root xlink without namespace
                    _svg_string = _svg_string.replace(/NS\d+:href/g, 'xlink:href'); // Safari NS namespace fix
                    return _svg_string;
                }
            });

            let _lineclick = (event) => {
                _clines.selectAll("line").classed("active", false);
                $(event.currentTarget).addClass("active");
            }

            let _create_line = (dotSource, dotTarget) => {
                let sourceNode = d3.select(dotSource.closest(".node")[0]),
                    sourceID = parseInt(dotSource.closest(".node")[0].dataset.index),
                    sourceMatrix = sourceNode._groups[0][0].transform.baseVal[0].matrix,
                    sourcePosition = [sourceMatrix.e, sourceMatrix.f+_LINE_HEIGHT/2+9];

                let targetNode = dotTarget ? d3.select(dotTarget.closest(".node")[0]) : undefined,
                    targetID = targetNode ? dotTarget.closest(".node")[0].dataset.index : "",
                    targetMatrix = targetNode ? targetNode._groups[0][0].transform.baseVal[0].matrix : undefined,
                    targetPosition = targetMatrix ? [targetMatrix.e, targetMatrix.f-_LINE_HEIGHT/2-9] : undefined;

                let line = _clines.append("line")
                    .attr("id", `line_${sourceID}_${targetID}`)
                    .attr("class", "line")
                    .attr("x1", sourcePosition[0])
                    .attr("y1", sourcePosition[1])
                    .attr("stroke", "#65B9F4")
                    .attr("stroke-width", 2);

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

            let _update_dots_per_node = (node) => {
                let data = node.data()[0];
                let clear_links = (items) => {
                    return items.filter((item) => {return item > 0});
                };
                data.down_link = clear_links(data.down_link);
                data.config.up_link = clear_links(data.config.up_link);
                node.select(".dot-source").attr("visibility", data.down_link.length ? "visible" : "hidden");
                node.select(".dot-target").attr("visibility", data.config.up_link.length ? "visible" : "hidden");
            }

            let _clear_links = (node) => {
                let data = node.data()[0],
                    index = parseInt(node._groups[0][0].dataset.index);
                data.down_link.forEach((id) => {
                    _remove_line($(`#line_${index}_${id}`));
                });
                data.config.up_link.forEach((id) => {
                    _remove_line($(`#line_${id}_${index}`));
                });
            }

            let _create_node = (index, layer) => {
                let w = _d3graph._groups[0][0].width.baseVal.value,
                    h = _d3graph._groups[0][0].height.baseVal.value,
                    layer_color = {input:"#FFB054",middle:"#89D764",output:"#8E51F2"};

                let node = _cnodes.append("g")
                    .attr("id", `node-${index}`)
                    .attr("data-index", index)
                    .attr("class", `node node-type-${layer.config.location_type}`);

                node.append("circle")
                    .attr("class", "dot dot-target")
                    .attr("visibility", "hidden")
                    .attr("cx", 0)
                    .attr("cy", -_NODE_HEIGHT/2-4)
                    .attr("r", 2)
                    .attr("fill", "#65B9F4");

                node.append("circle")
                    .attr("class", "dot dot-source")
                    .attr("visibility", "hidden")
                    .attr("cx", 0)
                    .attr("cy", _NODE_HEIGHT/2+4)
                    .attr("r", 2)
                    .attr("fill", "#65B9F4");

                let tools = node.append("g").attr("class", "tools"),
                    tools_rect = tools.append("rect").attr("class", "bg").attr("width", 92).attr("height", _NODE_HEIGHT+2).attr("y", -_NODE_HEIGHT/2-1),
                    rect = node.append("rect").attr("class", "node-bg").attr("height", _NODE_HEIGHT).attr("stroke-width", 2).attr("rx", 4).attr("ry", 4).attr("fill", layer_color[layer.config.location_type]).attr("stroke", layer_color[layer.config.location_type]),
                    text = node.append("text").attr("font-family", "Open Sans"),
                    text_name = text.append("tspan").text(`${layer.config.name}: ${layer.config.type}`).attr("class", "name").attr("y", -2).attr("fill", "#17212B").attr("font-size", "14px");
                text_name.attr("x", -text_name._groups[0][0].getBBox().width/2);

                let params_list = [];
                for (let param_name in layer.config.params.main) {
                    let value = `${layer.config.params.main[param_name]}`;
                    if (value !== '') {
                        if (value.indexOf(",") !== -1) value = `[${value}]`;
                        params_list.push(value);
                    }
                }
                if (params_list.length) {
                    let text_params = text.append("tspan").text(params_list.join(", ")).attr("class", "params").attr("y", 12).attr("fill", "#2B5278").attr("font-size", "11px");
                    text_params.attr("x", -text_params._groups[0][0].getBBox().width/2);
                } else {

                }

                let text_box = text._groups[0][0].getBBox(),
                    width = text_box.width + 20;

                let link_group = tools.append("g").attr("class", "btn link"),
                    link_icon = link_group.append("path").attr("width", 18).attr("height", 18).attr("fill", "#fff").attr("d", "M6.54114 12.4546C6.2768 12.4522 6.02373 12.347 5.83541 12.1613C5.65 11.9729 5.54606 11.7191 5.54606 11.4546C5.54606 11.1902 5.65 10.9363 5.83541 10.748L10.7356 5.84162C10.9249 5.66499 11.1753 5.56883 11.434 5.5734C11.6928 5.57797 11.9396 5.68292 12.1226 5.86612C12.3056 6.04933 12.4104 6.29649 12.4149 6.55555C12.4195 6.8146 12.3235 7.06532 12.1471 7.25487L7.24687 12.1613C7.15445 12.2544 7.04448 12.3283 6.92335 12.3787C6.80221 12.429 6.6723 12.4548 6.54114 12.4546ZM6.26151 17.7076L10.3827 13.5812C10.5337 13.3899 10.6096 13.1499 10.5963 12.9065C10.5829 12.6631 10.4812 12.4329 10.3103 12.2593C10.1393 12.0857 9.91092 11.9806 9.66801 11.9638C9.42511 11.9469 9.18443 12.0195 8.99123 12.1679L5.57575 15.5877L2.41327 12.4412L5.82875 9.00144C5.92687 8.90989 6.00557 8.7995 6.06015 8.67684C6.11474 8.55418 6.14409 8.42177 6.14645 8.28751C6.14882 8.15325 6.12415 8.01988 6.07392 7.89537C6.0237 7.77086 5.94893 7.65776 5.8541 7.5628C5.75927 7.46785 5.64631 7.393 5.52195 7.3427C5.3976 7.29241 5.26441 7.26771 5.13031 7.27008C4.99622 7.27245 4.86398 7.30184 4.74147 7.35649C4.61897 7.41114 4.50871 7.48994 4.41729 7.58819L0.296071 11.7346C0.202668 11.827 0.12851 11.937 0.0778966 12.0583C0.027283 12.1796 0.0012207 12.3098 0.0012207 12.4412C0.0012207 12.5727 0.027283 12.7029 0.0778966 12.8242C0.12851 12.9454 0.202668 13.0555 0.296071 13.1479L4.85005 17.7076C5.0373 17.8949 5.29113 18 5.55578 18C5.82043 18 6.07426 17.8949 6.26151 17.7076ZM13.5852 10.3747L17.7064 6.24826C17.7998 6.15588 17.8739 6.04584 17.9245 5.92455C17.9752 5.80325 18.0012 5.6731 18.0012 5.54164C18.0012 5.41018 17.9752 5.28003 17.9245 5.15873C17.8739 5.03743 17.7998 4.9274 17.7064 4.83501L13.1324 0.29527C13.0402 0.201748 12.9303 0.127496 12.8091 0.0768187C12.688 0.026141 12.558 4.57764e-05 12.4267 4.57764e-05C12.2954 4.57764e-05 12.1654 0.026141 12.0443 0.0768187C11.9231 0.127496 11.8132 0.201748 11.721 0.29527L7.59974 4.4217C7.48741 4.50798 7.39469 4.61718 7.32772 4.74206C7.26075 4.86695 7.22106 5.00466 7.21129 5.14607C7.20151 5.28748 7.22188 5.42936 7.27103 5.56229C7.32019 5.69522 7.39701 5.81616 7.4964 5.91711C7.59579 6.01805 7.71547 6.09668 7.84752 6.14779C7.97957 6.19889 8.12096 6.22131 8.26232 6.21354C8.40367 6.20577 8.54177 6.168 8.66743 6.10273C8.7931 6.03746 8.90348 5.94618 8.99123 5.83495L12.4067 2.41515L15.5492 5.56164L12.1537 9.00144C12.0603 9.09383 11.9862 9.20386 11.9355 9.32516C11.8849 9.44646 11.8589 9.57661 11.8589 9.70807C11.8589 9.83952 11.8849 9.96968 11.9355 10.091C11.9862 10.2123 12.0603 10.3223 12.1537 10.4147C12.341 10.6019 12.5948 10.7071 12.8594 10.7071C13.1241 10.7071 13.3779 10.6019 13.5652 10.4147L13.5852 10.3747Z"),
                    link = link_group.append("rect").attr("width", 18).attr("height", 18),
                    unlink_group = tools.append("g").attr("class", "btn unlink"),
                    unlink_icon_1 = unlink_group.append("path").attr("width", 18).attr("height", 18).attr("fill", "#fff").attr("d", "M6.50512 12.4545C6.24218 12.4522 5.99046 12.347 5.80314 12.1612C5.61872 11.9729 5.51534 11.719 5.51534 11.4546C5.51534 11.1901 5.61872 10.9363 5.80314 10.748L10.6773 5.84157C10.8656 5.66494 11.1146 5.56878 11.372 5.57336C11.6293 5.57793 11.8749 5.68287 12.0569 5.86608C12.2389 6.04928 12.3431 6.29645 12.3477 6.5555C12.3522 6.81456 12.2567 7.06527 12.0812 7.25483L7.2071 12.1612C7.11516 12.2544 7.00578 12.3283 6.88529 12.3786C6.7648 12.429 6.63558 12.4548 6.50512 12.4545ZM6.22698 17.7076L10.3263 13.5811C10.4764 13.3898 10.5519 13.1499 10.5387 12.9065C10.5254 12.663 10.4242 12.4329 10.2542 12.2593C10.0842 12.0857 9.85697 11.9805 9.61536 11.9637C9.37374 11.9469 9.13434 12.0195 8.94218 12.1679L5.54487 15.5877L2.39921 12.4412L5.79652 9.00139C5.89412 8.90985 5.9724 8.79945 6.02669 8.6768C6.08099 8.55414 6.11018 8.42173 6.11253 8.28746C6.11489 8.1532 6.09035 8.01984 6.04039 7.89533C5.99043 7.77082 5.91607 7.65771 5.82174 7.56276C5.72741 7.46781 5.61505 7.39295 5.49136 7.34266C5.36767 7.29237 5.23518 7.26767 5.1018 7.27004C4.96842 7.27241 4.83688 7.30179 4.71503 7.35645C4.59318 7.4111 4.48351 7.4899 4.39257 7.58814L0.293282 11.7346C0.200376 11.827 0.126612 11.937 0.0762679 12.0583C0.0259236 12.1796 0 12.3097 0 12.4412C0 12.5727 0.0259236 12.7028 0.0762679 12.8241C0.126612 12.9454 0.200376 13.0554 0.293282 13.1478L4.82302 17.7076C5.00928 17.8948 5.26176 18 5.525 18C5.78824 18 6.04072 17.8948 6.22698 17.7076ZM13.5117 10.3746L17.6109 6.24822C17.7038 6.15583 17.7776 6.0458 17.828 5.9245C17.8783 5.8032 17.9042 5.67305 17.9042 5.54159C17.9042 5.41013 17.8783 5.27998 17.828 5.15868C17.7776 5.03738 17.7038 4.92735 17.6109 4.83496L13.0613 0.295224C12.9695 0.201703 12.8602 0.127451 12.7397 0.0767729C12.6192 0.0260952 12.4899 0 12.3594 0C12.2288 0 12.0995 0.0260952 11.979 0.0767729C11.8585 0.127451 11.7492 0.201703 11.6574 0.295224L7.55809 4.42165C7.44636 4.50793 7.35413 4.61713 7.28751 4.74202C7.2209 4.8669 7.18142 5.00461 7.1717 5.14603C7.16198 5.28744 7.18224 5.42931 7.23113 5.56224C7.28002 5.69517 7.35643 5.81612 7.45529 5.91706C7.55415 6.01801 7.6732 6.09664 7.80455 6.14774C7.93589 6.19885 8.07653 6.22126 8.21714 6.21349C8.35774 6.20572 8.4951 6.16795 8.6201 6.10268C8.7451 6.03741 8.85489 5.94614 8.94218 5.83491L12.3395 2.4151L15.4653 5.56159L12.0878 9.00139C11.9949 9.09378 11.9212 9.20381 11.8708 9.32511C11.8205 9.44641 11.7945 9.57656 11.7945 9.70802C11.7945 9.83948 11.8205 9.96963 11.8708 10.0909C11.9212 10.2122 11.9949 10.3223 12.0878 10.4146C12.2741 10.6019 12.5266 10.7071 12.7898 10.7071C13.0531 10.7071 13.3055 10.6019 13.4918 10.4146L13.5117 10.3746Z"),
                    unlink_icon_2 = unlink_group.append("path").attr("width", 18).attr("height", 18).attr("fill", "#fff").attr("d", "M4.26487 5.40829C4.26487 5.14309 4.16021 4.88875 3.97392 4.70123C3.78763 4.5137 3.53496 4.40835 3.2715 4.40835H1.51656C1.2531 4.40835 1.00044 4.5137 0.814143 4.70123C0.627851 4.88875 0.523193 5.14309 0.523193 5.40829C0.523193 5.67349 0.627851 5.92783 0.814143 6.11536C1.00044 6.30289 1.2531 6.40824 1.51656 6.40824H3.2715C3.53496 6.40824 3.78763 6.30289 3.97392 6.11536C4.16021 5.92783 4.26487 5.67349 4.26487 5.40829ZM6.67543 3.00176V1.2152C6.67543 0.949998 6.57078 0.695658 6.38448 0.508132C6.19819 0.320607 5.94553 0.215256 5.68207 0.215256C5.41861 0.215256 5.16595 0.320607 4.97965 0.508132C4.79336 0.695658 4.6887 0.949998 4.6887 1.2152V3.00176C4.6887 3.26697 4.79336 3.5213 4.97965 3.70883C5.16595 3.89636 5.41861 4.00171 5.68207 4.00171C5.94553 4.00171 6.19819 3.89636 6.38448 3.70883C6.57078 3.5213 6.67543 3.26697 6.67543 3.00176ZM17.9998 12.6345C17.9998 12.3693 17.8951 12.115 17.7088 11.9275C17.5226 11.74 17.2699 11.6346 17.0064 11.6346H15.2316C14.9682 11.6346 14.7155 11.74 14.5292 11.9275C14.3429 12.115 14.2383 12.3693 14.2383 12.6345C14.2383 12.8998 14.3429 13.1541 14.5292 13.3416C14.7155 13.5291 14.9682 13.6345 15.2316 13.6345H16.9866C17.1204 13.6372 17.2533 13.6127 17.3775 13.5624C17.5016 13.5121 17.6144 13.4371 17.709 13.3418C17.8037 13.2466 17.8782 13.133 17.9282 13.0081C17.9781 12.8831 18.0025 12.7492 17.9998 12.6146V12.6345ZM13.8343 16.8276V15.0411C13.8343 14.7759 13.7296 14.5215 13.5433 14.334C13.357 14.1465 13.1044 14.0411 12.8409 14.0411C12.5775 14.0411 12.3248 14.1465 12.1385 14.334C11.9522 14.5215 11.8476 14.7759 11.8476 15.0411V16.8076C11.8476 17.0728 11.9522 17.3272 12.1385 17.5147C12.3248 17.7022 12.5775 17.8076 12.8409 17.8076C13.1044 17.8076 13.357 17.7022 13.5433 17.5147C13.7296 17.3272 13.8343 17.0728 13.8343 16.8076V16.8276Z"),
                    unlink = unlink_group.append("rect").attr("width", 18).attr("height", 18),
                    remove_group = tools.append("g").attr("class", "btn remove"),
                    remove_icon = remove_group.append("path").attr("width", 18).attr("height", 18).attr("fill", "#fff").attr("d", "M15.1528 3.2H1.05717C0.776794 3.2 0.507897 3.30536 0.309639 3.49289C0.11138 3.68043 0 3.93478 0 4.2C0 4.46522 0.11138 4.71957 0.309639 4.90711C0.507897 5.09464 0.776794 5.2 1.05717 5.2H1.40957V14.5867C1.41143 15.4914 1.7922 16.3586 2.46852 16.9983C3.14484 17.6381 4.06159 17.9982 5.01805 18H11.1919C12.1484 17.9982 13.0652 17.6381 13.7415 16.9983C14.4178 16.3586 14.7986 15.4914 14.8004 14.5867V5.2H15.1528C15.4332 5.2 15.7021 5.09464 15.9004 4.90711C16.0986 4.71957 16.21 4.46522 16.21 4.2C16.21 3.93478 16.0986 3.68043 15.9004 3.49289C15.7021 3.30536 15.4332 3.2 15.1528 3.2V3.2ZM12.6861 14.5867C12.6861 14.7723 12.6474 14.9561 12.5724 15.1275C12.4973 15.299 12.3872 15.4548 12.2485 15.586C12.1097 15.7173 11.945 15.8214 11.7637 15.8924C11.5825 15.9634 11.3882 16 11.1919 16H5.01805C4.62178 16 4.24174 15.8511 3.96154 15.586C3.68133 15.321 3.52391 14.9615 3.52391 14.5867V5.2H12.6861V14.5867ZM4.08774 1C4.08774 0.734784 4.19912 0.48043 4.39738 0.292893C4.59564 0.105357 4.86453 0 5.14491 0H11.0651C11.3455 0 11.6144 0.105357 11.8126 0.292893C12.0109 0.48043 12.1223 0.734784 12.1223 1C12.1223 1.26522 12.0109 1.51957 11.8126 1.70711C11.6144 1.89464 11.3455 2 11.0651 2H5.14491C4.86453 2 4.59564 1.89464 4.39738 1.70711C4.19912 1.51957 4.08774 1.26522 4.08774 1ZM5.00396 13.2667V7.93333C5.00396 7.66812 5.11534 7.41376 5.3136 7.22623C5.51185 7.03869 5.78075 6.93333 6.06113 6.93333C6.34151 6.93333 6.61041 7.03869 6.80866 7.22623C7.00692 7.41376 7.1183 7.66812 7.1183 7.93333V13.2667C7.1183 13.5319 7.00692 13.7862 6.80866 13.9738C6.61041 14.1613 6.34151 14.2667 6.06113 14.2667C5.78075 14.2667 5.51185 14.1613 5.3136 13.9738C5.11534 13.7862 5.00396 13.5319 5.00396 13.2667ZM9.09169 13.2667V7.93333C9.09169 7.66812 9.20308 7.41376 9.40133 7.22623C9.59959 7.03869 9.86849 6.93333 10.1489 6.93333C10.4292 6.93333 10.6981 7.03869 10.8964 7.22623C11.0947 7.41376 11.206 7.66812 11.206 7.93333V13.2667C11.206 13.5319 11.0947 13.7862 10.8964 13.9738C10.6981 14.1613 10.4292 14.2667 10.1489 14.2667C9.86849 14.2667 9.59959 14.1613 9.40133 13.9738C9.20308 13.7862 9.09169 13.5319 9.09169 13.2667Z"),
                    remove = remove_group.append("rect").attr("width", 18).attr("height", 18);

                let rect_pointer = node.append("rect")
                    .attr("class", "pointer")
                    .attr("height", _NODE_HEIGHT+2)
                    .attr("rx", 4)
                    .attr("ry", 4)
                    .call(d3.drag()
                        .on("drag", _node_dragged)
                        .on("end", _node_dragended)
                    );

                $(rect_pointer._groups[0][0]).bind("mouseup", (event) => {
                    if (event.button === 2 && !_new_link) {
                        $(event.currentTarget).closest(".node").find(".tools > .link").trigger("click");
                    }
                });

                rect.attr("width", width).attr("x", -(text_box.width+20)/2).attr("y", -rect._groups[0][0].height.baseVal.value/2);
                rect_pointer.attr("width", width+5).attr("x", -(text_box.width+20)/2-1).attr("y", -rect._groups[0][0].height.baseVal.value/2-1);
                tools_rect.attr("x", (text_box.width+20)/2+3);
                link_group.attr("transform", `translate(${(text_box.width+20)/2+13}, -9)`);
                unlink_group.attr("transform", `translate(${(text_box.width+20)/2+41}, -9)`);
                remove_group.attr("transform", `translate(${(text_box.width+20)/2+69}, -9)`);

                $(link_group._groups[0][0]).bind("click", (event) => {
                    this.addClass("onlink");
                    _create_line($(event.currentTarget).closest(".node").children(".dot-source"));
                });
                $(unlink_group._groups[0][0]).bind("click", (event) => {
                    let node = _d3graph.select(`#${$(event.currentTarget).closest(".node")[0].id}`);
                    _clear_links(node);
                    terra_toolbar.btn.save.disabled = false;
                    $(terra_toolbar.btn.save).children("span").trigger("click");
                });
                $(remove_group._groups[0][0]).bind("click", (event) => {
                    let g = $(event.currentTarget).closest(".node"),
                        index = parseInt(g[0].dataset.index),
                        node = _d3graph.select(`#${g[0].id}`);
                    _clear_links(node);
                    g.remove();
                    if (`${index}` === `${$("#field_form-index").val()}`) terra_params.reset();
                    let layers = window.TerraProject.layers;
                    delete layers[index];
                    window.TerraProject.layers = layers;
                    terra_toolbar.btn.save.disabled = false;
                    $(terra_toolbar.btn.save).children("span").trigger("click");
                });

                if (["input", "output"].indexOf(layer.config.location_type) > -1) {
                    tools_rect.attr("width", 66);
                    remove_group.remove();
                }

                if (layer.x === null) layer.x = w/2;
                if (layer.y === null) layer.y = h/2;

                node.data([layer])
                    .attr("transform", "translate(" + layer.x + "," + layer.y + ")");

                $(node._groups[0][0]).bind("mouseenter", (event) => {
                    _onNode = $(event.currentTarget);
                    _d3graph.select(`#${_onNode[0].id}`).raise();
                }).bind("mouseleave", (event) => {
                    if (_new_link) {
                        let node = _d3graph.select(`#${event.currentTarget.id}`),
                            data = node.data()[0],
                            sourceID = _new_link._groups[0][0].sourceID;
                        if (`${sourceID}` !== `${node._groups[0][0].dataset.index}` && data.config.up_link.indexOf(sourceID) === -1) {
                            _update_dots_per_node(node);
                        }
                    }
                    _onNode = undefined;
                });

                terra_toolbar.btn.save.disabled = false;

            };

            let _create_model = (layers, schema) => {
                let w = _d3graph._groups[0][0].width.baseVal.value,
                    h = _d3graph._groups[0][0].height.baseVal.value;

                _clines.selectAll("line").remove();
                _cnodes.selectAll("g").remove();
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

                let use_schema = false;
                for (let index in layers) {
                    let layer = layers[index];
                    if ((layer.x === null || layer.y === null) && ["input", "output"].indexOf(layer.config.location_type) > -1) use_schema = true;
                    _create_node(parseInt(index), layer);
                }
                if (use_schema && schema.length) _update_position_by_schema();

                for (let index in layers) {
                    let layer = layers[index];
                    layer.config.up_link.forEach((item) => {
                        if (item !== 0) {
                            _create_line($(`#node-${item} > .dot-source`), $(`#node-${index} > .dot-target`));
                        }
                    });
                }

                window.TerraProject.layers = layers;
                $(terra_toolbar.btn.save).children("span").trigger("click");
            }

            this.clear = () => {
                window.StatusBar.clear();
                ClearModel.open();
            }

            return this;

        },


        TerraParams: function() {

            if (!this.length) return this;

            let _layer_index_field = $("#field_form-index"),
                _layer_name_field = $("#field_form-name"),
                _layer_type_field = $("#field_form-type"),
                _layer_params_main = this.find(".params-main"),
                _layer_params_extra = this.find(".params-extra"),
                _action_save = this.find(".actions-form > .item.save > button"),
                _action_clone = this.find(".actions-form > .item.clone > button"),
                _on_change_param_timer;

            let _camelize = (text) => {
                let _capitalize = (word) => {
                    return `${word.slice(0, 1).toUpperCase()}${word.slice(1).toLowerCase()}`
                }
                let words = text.split("_"),
                    result = [_capitalize(words[0])];
                words.slice(1).forEach((word) => result.push(word))
                return result.join(" ")
            }

            let _on_change_params = () => {
                if (_on_change_param_timer) clearTimeout(_on_change_param_timer);
                _on_change_param_timer = setTimeout(() => {
                    this.trigger("submit", {"no_close":true});
                }, 500);
            }

            let _render_params = (config) => {
                this.find("#field_form-input_shape").parent().remove();
                this.find("#field_form-data_name").parent().remove();
                _layer_params_main.addClass("hidden");
                _layer_params_main.children(".inner").html("");
                _layer_params_extra.addClass("hidden");
                _layer_params_extra.children(".inner").html("");
                let _render_params_config = (group, container, params, data) => {
                    let inner = container.children(".inner");
                    if (!Object.keys(params).length) return;
                    for (let name in params) {
                        let param = $.extend(true, {}, params[name]);
                        param.label = name;
                        if (data[name] !== undefined) param.default = data[name];
                        param.label = _camelize(name);
                        let widget = window.FormWidget(`params[${group}][${name}]`, param);
                        widget.find("input").bind("input change", _on_change_params);
                        widget.find("select").selectmenu({
                            change:(event, ui) => {
                                $(event.target).trigger("change");
                            }
                        }).bind("change", _on_change_params);
                        widget.addClass("field-inline");
                        inner.append(widget);
                    }
                    container.removeClass("hidden");
                }
                let params_config = window.TerraProject.layers_types[config.type];
                _render_params_config("main", _layer_params_main, params_config.main, config.params.main);
                _render_params_config("extra", _layer_params_extra, params_config.extra, config.params.extra);
                if (config.data_available && config.data_available.length) {
                    let WidgetData = window.FormWidget("data_name", {
                        "label":"Данные слоя",
                        "type":"str",
                        "list":true,
                        "default":config.data_name,
                        "available":config.data_available,
                        "disabled":true,
                    });
                    WidgetData.find("input").bind("input", _on_change_params);
                    this.find(".params-config > .inner").append(WidgetData);
                }
                if (config.location_type === "input") {
                    let WidgetInputShape = window.FormWidget("input_shape", {
                        "label":"Размерность входа",
                        "type":"tuple",
                        "list":false,
                        "default":config.input_shape,
                        "disabled":true,
                    });
                    WidgetInputShape.find("input").bind("input", _on_change_params);
                    this.find(".params-config > .inner").append(WidgetInputShape);
                }
            }

            this.reset = () => {
                _layer_index_field.val("");
                _layer_name_field.val("").attr("disabled", "disabled");
                _layer_type_field.val("").attr("disabled", "disabled");
                this.find("#field_form-input_shape").parent().remove();
                this.find("#field_form-data_name").parent().remove();
                _action_save.attr("disabled", "disabled");
                _action_clone.attr("disabled", "disabled");
                _layer_params_main.addClass("hidden");
                _layer_params_main.children(".inner").html("");
                _layer_params_extra.addClass("hidden");
                _layer_params_extra.children(".inner").html("");
            }

            this.load = (index, data) => {
                this.reset();
                _layer_index_field.val(index);
                _layer_name_field.val(data.config.name).removeAttr("disabled");
                _layer_type_field.val(data.config.type);
                if (data.config.location_type !== "input") _layer_type_field.removeAttr("disabled");
                _action_save.removeAttr("disabled");
                if (data.config.location_type === "middle") _action_clone.removeAttr("disabled");
                _render_params(data.config);
            }

            let _prepare_data = (_form) => {
                let _config = $.extend(true, {}, terra_board.find(`#node-${_form.index}`)[0].__data__),
                    _params = window.TerraProject.layers_types[_form.type];

                _config.config.name = _form.name;
                _config.config.type = _layer_type_field.val();
                _config.config.data_name = _form.data_name || "";
                _config.config.params = {"main":{},"extra":{}};

                for (let group in _params) {
                    for (let name in _params[group]) {
                        let value = _form.params[group][name];
                        switch (_params[group][name].type) {
                            case "int":
                                value = parseInt(value);
                                break;
                            case "float":
                                value = parseFloat(value);
                                break;
                            case "bool":
                                value = value !== undefined;
                                break;
                        }
                        _config.config.params[group][name] = value
                    }
                }

                return _config;
            };

            _layer_name_field.bind("input", _on_change_params);

            _layer_type_field.bind("blur", (event) => {
                let _config = $.extend(true, {}, terra_board.find(`#node-${_layer_index_field.val()}`)[0].__data__);
                _render_params({
                    "type":event.currentTarget.value,
                    "data_name":_config.config.data_name,
                    "data_available":_config.config.data_available,
                    "params":{"main":{},"extra":{}},
                });
                _on_change_params();
            });

            _action_clone.bind("click", (event) => {
                event.preventDefault();
                window.StatusBar.clear();
                let indexes = Object.keys(window.TerraProject.layers).map((value) => {
                    return parseInt(value);
                });
                if (!indexes.length) indexes = [0];
                let _max_id = Math.max.apply(Math, indexes)+1,
                    layer = _prepare_data($(event.currentTarget).closest("form").serializeObject());
                layer.config.name = `L${_max_id}`;
                layer.config.up_link = [];
                layer.down_link = [];
                layer.x = null;
                layer.y = null;
                terra_toolbar.btn.save_model.disabled = true;
                terra_toolbar.btn.keras.disabled = true;
                window.ExchangeRequest(
                    "save_layer",
                    (success, data) => {
                        if (success) {
                            terra_board.model = {"layers":data.data.layers,"schema":[]};
                            window.StatusBar.message(window.Messages.get("LAYER_CLONED"), true);
                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    },
                    {
                        "index": _max_id,
                        "layer": layer,
                    }
                );
            });

            this.bind("submit", (event, data) => {
                let _no_close = data !== undefined && data.no_close,
                    _layer_index = parseInt(_layer_index_field.val());
                if (_on_change_param_timer) clearTimeout(_on_change_param_timer);
                event.preventDefault();
                window.StatusBar.clear();
                terra_toolbar.btn.save_model.disabled = true;
                terra_toolbar.btn.keras.disabled = true;
                window.ExchangeRequest(
                    "save_layer",
                    (success, data) => {
                        if (success) {
                            terra_board.model = {"layers":data.data.layers,"schema":[],"no_close":_no_close};
                            if (_no_close) terra_board.activeNode($(`#node-${_layer_index}`)[0], _no_close);
                            window.StatusBar.message(window.Messages.get("LAYER_SAVED"), true);
                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    },
                    {
                        "index": _layer_index,
                        "layer": _prepare_data($(event.currentTarget).serializeObject()),
                    }
                );
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
            if ($.cookie("model_need_validation")) {
                $(terra_toolbar.btn.validation).children("span").trigger("click");
                $.removeCookie("model_need_validation", {path:window.TerraProject.path.modeling});
            }
        }

        LoadModel.find(".model-save-arch-btn > button").bind("click", (event) => {
            window.StatusBar.clear();
            window.ExchangeRequest(
                "set_model",
                (success, data) => {
                    if (success) {
                        window.TerraProject.layers = data.data.layers;
                        window.TerraProject.layers_schema = data.data.schema;
                        terra_board.model = window.TerraProject.model_info;
                        terra_toolbar.btn.save_model.disabled = true;
                        terra_toolbar.btn.keras.disabled = true;
                        LoadModel.close();
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
                {
                    "layers": event.currentTarget.ModelData.layers,
                    "schema": event.currentTarget.ModelData.schema,
                    "reset_training": true
                }
            )
        });

        KerasCode.find(".action > .clipboard").bind("click", (event) => {
            let result = KerasCode.find(".action > .result"),
                pre = KerasCode.find("pre");
            result.text("");
            if (fallbackCopyTextToClipboard(pre.text())) {
                result.text("Код скопирован в буфер обмена");
                clip(pre[0]);
            }
        });

        SaveModel.find("form").bind("submit", (event) => {
            event.preventDefault();
            let data = $(event.currentTarget).serializeObject();
            data.overwrite = data.overwrite !== undefined;
            data.preview = terra_board.svg;
            window.ExchangeRequest(
                "save_model",
                (success, data) => {
                    if (success) {
                        SaveModel.close();
                        window.TerraProject.model_name = data.data.name;
                        window.StatusBar.message(window.Messages.get("MODEL_SAVED"), true);
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
                data
            );
        });

        ClearModel.find("form").bind("submit", (event) => {
            event.preventDefault();
            window.ExchangeRequest(
                "clear_model",
                (success, data) => {
                    if (success) {
                        ClearModel.close();
                        terra_board.model = data.data;
                        terra_toolbar.btn.save.disabled = true;
                        terra_toolbar.btn.save_model.disabled = true;
                        terra_toolbar.btn.keras.disabled = true;
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
            )
        });

        ClearModel.find(".actions-form > .cancel > button").bind("click", (event) => {
            event.preventDefault();
            ClearModel.close();
        });

    });

})(jQuery);

