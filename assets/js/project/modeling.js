"use strict";


(($) => {


    function getSVGString( svgNode ) {
        svgNode.setAttribute('xlink', 'http://www.w3.org/1999/xlink');

        var serializer = new XMLSerializer();
        var svgString = serializer.serializeToString(svgNode);
        svgString = svgString.replace(/(\w+)?:?xlink=/g, 'xmlns:xlink='); // Fix root xlink without namespace
        svgString = svgString.replace(/NS\d+:href/g, 'xlink:href'); // Safari NS namespace fix

        return svgString;
    }
    function svgString2Image( svgString, width, height, format, callback ) {
        var format = format ? format : 'png';

        var imgsrc = 'data:image/svg+xml;base64,'+ btoa( unescape( encodeURIComponent( svgString ) ) ); // Convert SVG string to data URL
        console.log(imgsrc);

        var canvas = document.createElement("canvas");
        var context = canvas.getContext("2d");

        canvas.width = width;
        canvas.height = height;

        var image = new Image();
        image.src = imgsrc;
        var out = canvas.toDataURL("image/png");
        image.onload = function() {
            context.clearRect ( 0, 0, width, height );
            context.drawImage(image, 0, 0, width, height);
            canvas.toBlob( function(blob) {
                var filesize = Math.round( blob.length/1024 ) + ' KB';
                if ( callback ) callback( blob, filesize );
            });
        };
        console.log(out);
        // send_data(out)
    }


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
            let map_replace = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&#34;',
                "'": '&#39;'
            };
            ui.find(".wrapper .content").html(`<code>${data.code.replace(/[&<>'"]/g, (c) => {return map_replace[c]})}</code>`);
        }
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
                save_model: (item, callback) => {
                    console.log(item, callback);
                    // let send_data = {};
                    // d3.selectAll("g.node")._groups[0].forEach((item) => {
                    //     send_data[parseInt(item.dataset.index)] = item.__data__;
                    // });
                    // window.StatusBar.clear();
                    // window.ExchangeRequest(
                    //     "set_model",
                    //     (success, data) => {
                    //         if (success) {
                    //             this.btn.save.disabled = true;
                    //             window.StatusBar.message(window.Messages.get("MODEL_SAVED"), true);
                    //             if (typeof callback === "function") callback(item);
                    //         } else {
                    //             window.StatusBar.message(data.error, false);
                    //         }
                    //     },
                    //     {
                    //         "layers": send_data,
                    //         "schema": window.TerraProject.schema,
                    //     }
                    // );
                },
                save: (item, callback) => {
                    let send_data = {};
                    d3.selectAll("g.node")._groups[0].forEach((item) => {
                        send_data[parseInt(item.dataset.index)] = item.__data__;
                    });
                    window.StatusBar.clear();
                    terra_toolbar.btn.save_model.disabled = true;
                    terra_toolbar.btn.keras.disabled = true;
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
                    window.ExchangeRequest(
                        "get_change_validation",
                        (success, data) => {
                            this.btn.validation.disabled = false;
                            // let svg = document.getElementsByTagName("svg")[0],
                            //     svg_string = getSVGString(svg);
                            // svgString2Image(
                            //     svg_string,
                            //     svg.width.baseVal.value,
                            //     svg.height.baseVal.value,
                            //     "png"
                            // );
                            // console.log(svg_string);
                            if (success) {
                                window.StatusBar.clear();
                                let is_error = false;
                                for (let index in data.data) {
                                    let error = data.data[index];
                                    if (error) {
                                        terra_board.set_layer_error(index, is_error ? "" : JSON.stringify(error));
                                        is_error = true;
                                    }
                                }
                                if (!is_error) {
                                    window.StatusBar.message(window.Messages.get("VALIDATION_MODEL_SUCCESS"), true);
                                    terra_toolbar.btn.save_model.disabled = false;
                                    terra_toolbar.btn.keras.disabled = false;
                                }
                            } else {
                                window.StatusBar.message(data.error, false);
                            }
                        }
                    );
                },
                keras: (item, callback) => {
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

            const _NODE_HEIGHT = 26,
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

            this.set_layer_error = (index, message) => {
                _cnodes.select(`#node-${index}`).classed("error", true);
                if (message) window.StatusBar.message(`[${window.TerraProject.layers[index].config.name}: ${window.TerraProject.layers[index].config.type}] - ${message}`, false);
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

                let indexes = Object.keys(window.TerraProject.layers).map((value) => {
                    return parseInt(value);
                });
                if (!indexes.length) indexes = [0];
                let _max_id = Math.max.apply(Math, indexes)+1;

                let layer_config = {
                    name: `l${_max_id}_${type}`,
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

            this.activeNode = (g) => {
                let node = _cnodes.select(`#${g.id}`);
                _cnodes.selectAll(".node").classed("active", false);
                node.classed("active", true);
                terra_params.load(parseInt(g.dataset.index), node.data()[0]);
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
                        y2 = matrix.f - _LINE_HEIGHT / 2 - 2;
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
                    _clines.select(`#line_${node.dataset.index}_${index}`).attr("x1", x).attr("y1", y+_LINE_HEIGHT/2+2);
                });
                info.config.up_link.forEach((index) => {
                    _clines.select(`#line_${index}_${node.dataset.index}`).attr("x2", x).attr("y2", y-_LINE_HEIGHT/2-2);
                });
            }

            let _node_dragended = (data, _, rect) => {
                let node = $(rect).parent()[0],
                    _node = d3.select(`#${node.id}`);
                if (_onDrag) {
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
                                .attr("y2", matrix.f - _LINE_HEIGHT / 2 - 2);
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
                    sourceID = parseInt(dotSource.closest(".node")[0].dataset.index),
                    sourceMatrix = sourceNode._groups[0][0].transform.baseVal[0].matrix,
                    sourcePosition = [sourceMatrix.e, sourceMatrix.f+_LINE_HEIGHT/2+2];

                let targetNode = dotTarget ? d3.select(dotTarget.closest(".node")[0]) : undefined,
                    targetID = targetNode ? dotTarget.closest(".node")[0].dataset.index : "",
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
                    h = _d3graph._groups[0][0].height.baseVal.value;

                let node = _cnodes.append("g")
                    .attr("id", `node-${index}`)
                    .attr("data-index", index)
                    .attr("class", `node node-type-${layer.config.location_type}`);

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

                let rect_pointer = node.append("rect")
                    .attr("class", "pointer")
                    .attr("height", _NODE_HEIGHT)
                    .call(d3.drag()
                        .on("drag", _node_dragged)
                        .on("end", _node_dragended)
                    );

                $(rect_pointer._groups[0][0]).bind("mouseup", (event) => {
                    if (event.button === 2 && !_new_link) {
                        $(event.currentTarget).closest(".node").find(".tools > .link").trigger("click");
                    }
                });

                text.attr("x", -text_box.width/2).attr("y", 12-text_box.height/2);
                rect.attr("width", width).attr("x", -(text_box.width+20)/2).attr("y", -rect._groups[0][0].height.baseVal.value/2);
                rect_pointer.attr("width", width).attr("x", -(text_box.width+20)/2).attr("y", -rect._groups[0][0].height.baseVal.value/2);
                tools_rect.attr("width", width).attr("x", -(text_box.width+20)/2);
                remove.attr("x", (text_box.width+20)/2-16);
                link.attr("x", -(text_box.width+20)/2+4);
                unlink.attr("x", -(text_box.width+20)/2+20);

                $(remove._groups[0][0]).bind("click", (event) => {
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

                if (["input", "output"].indexOf(layer.config.location_type) > -1) remove.remove();

                if (layer.x === null) layer.x = w/2;
                if (layer.y === null) layer.y = h/2;

                node.data([layer])
                    .attr("transform", "translate(" + layer.x + "," + layer.y + ")");

                $(node._groups[0][0]).bind("mouseenter", (event) => {
                    _onNode = $(event.currentTarget);
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
                terra_toolbar.btn.clear.disabled = false;

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
            _action_clone = this.find(".actions-form > .item.clone > button");

            let _camelize = (text) => {
                let _capitalize = (word) => {
                    return `${word.slice(0, 1).toUpperCase()}${word.slice(1).toLowerCase()}`
                }
                let words = text.split("_"),
                    result = [_capitalize(words[0])];
                words.slice(1).forEach((word) => result.push(word))
                return result.join(" ")
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
                    this.find(".params-config > .inner").append(WidgetInputShape);
                }
            }

            this.reset = () => {
                _layer_index_field.val("");
                _layer_name_field.val("").attr("disabled", "disabled");
                _layer_type_field.val("").attr("disabled", "disabled").selectmenu("refresh");
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
                _layer_type_field.selectmenu("refresh");
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

                for (let group in _params) {
                    for (let name in _params[group]) {
                        let value = _form.params[group][name];
                        switch (_params[group][name].type) {
                            case "bool":
                                value = value !== undefined;
                                break;
                        }
                        _config.config.params[group][name] = value
                    }
                }

                return _config;
            };

            _layer_type_field.bind("change", (event) => {
                let _config = $.extend(true, {}, terra_board.find(`#node-${_layer_index_field.val()}`)[0].__data__);
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

            _action_clone.bind("click", (event) => {
                event.preventDefault();
                window.StatusBar.clear();
                let indexes = Object.keys(window.TerraProject.layers).map((value) => {
                    return parseInt(value);
                });
                if (!indexes.length) indexes = [0];
                let _max_id = Math.max.apply(Math, indexes)+1,
                    layer = _prepare_data($(event.currentTarget).closest("form").serializeObject());
                layer.config.name = `l${_max_id}_${layer.config.type}`;
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

            this.bind("submit", (event) => {
                event.preventDefault();
                window.StatusBar.clear();
                terra_toolbar.btn.save_model.disabled = true;
                terra_toolbar.btn.keras.disabled = true;
                window.ExchangeRequest(
                    "save_layer",
                    (success, data) => {
                        if (success) {
                            terra_board.model = {"layers":data.data.layers,"schema":[]};
                            window.StatusBar.message(window.Messages.get("LAYER_SAVED"), true);
                        } else {
                            window.StatusBar.message(data.error, false);
                        }
                    },
                    {
                        "index": parseInt(_layer_index_field.val()),
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
                        LoadModel.close();
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
                {
                    "layers": event.currentTarget.ModelData.layers,
                    "schema": event.currentTarget.ModelData.front_model_schema,
                }
            )
        });

    });

})(jQuery);

