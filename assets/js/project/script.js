"use strict";


(($) => {


    window.Logging = $("#modal-window-logging").ModalWindow({
        title:"Logging",
        width:900,
        height:600,
        no_clear_status_bar:true,
    });


    let ProjectNew = $("#modal-window-project-new").ModalWindow({
        title:"Создать новый проект",
        width:300,
        height:190
    });


    let ProjectSave = $("#modal-window-project-save").ModalWindow({
        title:"Сохранить проект",
        width:400,
        height:212
    });


    let ProjectLoad = $("#modal-window-project-load").ModalWindow({
        title:"Загрузить проект",
        width:400,
        height:440,
        request:["project_load"]
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


    let TerraProject = function(hash) {

        const DEFAULT_NAME = "NoName",
            DEFAULT_HARDWARE = "CPU";

        let options = {},
            hash_decoded = $.base64.atob(hash);
        try {
            options = JSON.parse(hash_decoded);
        } catch (e) {
            console.warn(window.Messages.get("PARSE_TERRA_PROJECT_CONFIG_ERROR"));
        }

        let _error = options.error || "";
        let _tensorflow = options.tensorflow;
        let _name = options.name || DEFAULT_NAME;
        let _hardware = options.hardware || DEFAULT_HARDWARE;
        let _datasets = options.datasets || {};
        let _tags = options.tags || {};
        let _dataset = options.dataset || "";
        let _model_name = options.model_name || "";
        let _model_validated = options.model_validated || false;
        let _layers = options.layers || {};
        let _layers_start = options.layers_start || {};
        let _layers_schema = options.layers_schema || [];
        let _layers_types = options.layers_types || {};
        let _optimizers = options.optimizers || {};
        let _callbacks = options.callbacks || {};
        let _compile = options.compile || {};
        let _training = options.training || {};
        let _in_training = options.in_training || false;
        let _path = options.path || {};
        let _logging = options.logging || "No log";

        this.model_clear = () => {
            _layers = {};
            _layers_schema = [];
        };

        this.dataset_exists = (dataset_name) => {
            return this.datasets[dataset_name] !== undefined;
        }

        this.exec = {
            project_new: () => {
                ProjectNew.open();
            },
            project_save: () => {
                ProjectSave.open((target) => {
                    $("#field_form-project_save_name").val(window.TerraProject.name).focus();
                });
            },
            project_load: () => {
                window.StatusBar.clear();
                ProjectLoad.open(
                    (block, data) => {
                        block.find(".models-data > .models-list .loaded-list").html("");
                        for (let index in data) {
                            block.find(".models-data > .models-list .loaded-list").append($(`<li data-name="${data[index]}"><span>${data[index]}</span></li>`));
                        }
                        block.find(".models-data > .models-list .loaded-list > li > span").bind("click", (event) => {
                            let item = $(event.currentTarget).parent();
                            item.parent().children("li").removeClass("active");
                            item.addClass("active");
                            window.ExchangeRequest(
                                "get_project",
                                (success, data) => {
                                    if (success) {
                                        window.location.reload();
                                    } else {
                                        window.StatusBar.message(data.error, false);
                                    }
                                },
                                {"name":item.data("name")}
                            );
                        });
                    }
                );
            }
        }

        Object.defineProperty(this, "error", {
            set: (value) => {
                _error = value;
            },
            get: () => {
                return _error;
            }
        });

        Object.defineProperty(this, "tensorflow", {
            get: () => {
                return _tensorflow;
            }
        });

        Object.defineProperty(this, "name", {
            set: (value) => {
                if (!_name) _name = DEFAULT_NAME;
                if (!value) value = _name;
                $("header > .project > .title > .name > .value > span").text(value);
                if (_name === value) return;
                window.StatusBar.message(window.Messages.get("TRY_TO_SAVE_PROJECT_NAME", [value]));
                window.ExchangeRequest(
                    "set_project_name",
                    (success, data) => {
                        if (success) {
                            _name = value;
                            window.StatusBar.message(window.Messages.get("PROJECT_NAME_SAVE_SUCCESS", [value]), true);
                        } else {
                            $("header > .project > .title > .name > .value > span").text(_name);
                            window.StatusBar.message(data.error, false);
                        }
                    },
                    {"name":value}
                );
            },
            get: () => {
                return _name;
            }
        });

        Object.defineProperty(this, "hardware", {
            set: (value) => {
                _hardware = value;
            },
            get: () => {
                return _hardware;
            }
        });

        Object.defineProperty(this, "datasets", {
            get: () => {
                let output = {};
                _datasets.forEach((item) => {
                    output[item.name] = item;
                });
                return output;
            }
        });

        Object.defineProperty(this, "tags", {
            set: (value) => {
                _tags = value;
            },
            get: () => {
                return _tags;
            }
        });

        Object.defineProperty(this, "dataset", {
            set: (value) => {
                _dataset = value;
            },
            get: () => {
                return _dataset;
            }
        });

        Object.defineProperty(this, "dataset_selected", {
            get: () => {
                return _dataset !== "";
            }
        });

        Object.defineProperty(this, "model_name", {
            set: (value) => {
                _model_name = value;
            },
            get: () => {
                return _model_name;
            }
        });

        Object.defineProperty(this, "model_validated", {
            set: (value) => {
                _model_validated = value;
            },
            get: () => {
                return _model_validated;
            }
        });

        Object.defineProperty(this, "layers", {
            set: (value) => {
                _layers = value;
            },
            get: () => {
                return _layers;
            }
        });

        Object.defineProperty(this, "layers_start", {
            set: (value) => {
                _layers_start = value;
            },
            get: () => {
                return _layers_start;
            }
        });

        Object.defineProperty(this, "layers_schema", {
            set: (value) => {
                _layers_schema = value;
            },
            get: () => {
                return _layers_schema;
            }
        });

        Object.defineProperty(this, "layers_types", {
            set: (value) => {
                _layers_types = value;
            },
            get: () => {
                return _layers_types;
            }
        });

        Object.defineProperty(this, "model_info", {
            get: () => {
                return {
                    "layers": _layers,
                    "schema": _layers_schema,
                    "start_layers": _layers_start,
                };
            }
        });

        Object.defineProperty(this, "optimizers", {
            set: (value) => {
                _optimizers = value;
            },
            get: () => {
                return _optimizers;
            }
        });

        Object.defineProperty(this, "callbacks", {
            set: (value) => {
                _callbacks = value;
            },
            get: () => {
                return _callbacks;
            }
        });

        Object.defineProperty(this, "compile", {
            set: (value) => {
                _compile = value;
            },
            get: () => {
                return _compile;
            }
        });

        Object.defineProperty(this, "training", {
            set: (value) => {
                _training = value;
            },
            get: () => {
                return _training;
            }
        });

        Object.defineProperty(this, "in_training", {
            set: (value) => {
                _in_training = value;
            },
            get: () => {
                return _in_training;
            }
        });

        Object.defineProperty(this, "path", {
            set: (value) => {
                _path = value;
            },
            get: () => {
                return _path;
            }
        });

        Object.defineProperty(this, "logging", {
            set: (value) => {
                _logging = value;
            },
            get: () => {
                return _logging;
            }
        });

    }


    $(() => {

        window.TerraProject = new TerraProject(window._terra_project);

        if (window.TerraProject.error) {
            window.StatusBar.message(window.TerraProject.error, false);
        }

        $("header > .user > .item > .menu > .group > .title").bind("click", (event) => {
            $(event.currentTarget).parent().toggleClass("hidden");
        });

        $(".params-item.collapsable > .params-title").bind("click", (event) => {
            event.preventDefault();
            $(event.currentTarget).parent().toggleClass("collapsed");
        });

        /**
         * Редактирование названия проекта
         */
        $("header > .project > .title > .name > .value > span").bind("mousedown", (event) => {
            let item = $(event.currentTarget);
            item.attr("contenteditable", "true");
            item.focusin();
        }).focusout((event) => {
            let item = $(event.currentTarget);
            item.attr("contenteditable", "false");
            let name = item.text();
            if (!name) name = window.TerraProject.name;
            window.TerraProject.name = name;
        }).bind("keydown", (event) => {
            if (event.keyCode !== 13) return;
            let item = $(event.currentTarget);
            item.attr("contenteditable", "false");
        });

        /**
         * Выпадающее меню пользовательского блока
         */
        $("header > .user > .item > .icon > i").bind("click", (event) => {
            event.preventDefault();
            let item = $(event.currentTarget).closest(".item");
            let all = $("header > .user > .item").not(item);
            all.removeClass("active");
            if (item.hasClass("project")) {
                window.TerraProject.exec[item.data("type")]();
            } else {
                item.toggleClass("active");
            }
        });
        $(document).bind("click", (event) => {
            let item = $(event.target);
            if (!item.closest(".user > .item").length) {
                $("header > .user > .item").removeClass("active");
            }
        });

        /**
         * Сохранение результата обучения
         */
        $(window).bind("beforeunload", (event) => {
            window.ExchangeRequest("autosave_project", null, null, true);
        });

        $(document).bind("keyup", (event) => {
            if (event.keyCode === 76 && event.altKey && event.ctrlKey && event.shiftKey) {
                let text = window.TerraProject.logging ? window.TerraProject.logging : "No log";
                window.Logging.open((ui) => {
                    ui.find(".action > .result").text("");
                    let map_replace = {
                        '&': '&amp;',
                        '<': '&lt;',
                        '>': '&gt;',
                        '"': '&#34;',
                        "'": '&#39;'
                    };
                    ui.find(".wrapper .content").html(`<pre>${text.replace(/[&<>'"]/g, (c) => {return map_replace[c]})}</pre>`);
                });
            }
        });

        window.Logging.find(".action > .clipboard").bind("click", (event) => {
            let result = window.Logging.find(".action > .result"),
                pre = window.Logging.find("pre");
            result.text("");
            if (fallbackCopyTextToClipboard(pre.text())) {
                result.text("Код скопирован в буфер обмена");
                clip(pre[0]);
            }
        });

        ProjectNew.find("form").bind("submit", (event) => {
            event.preventDefault();
            window.ExchangeRequest(
                "project_new",
                (success, data) => {
                    if (success) {
                        window.location.href = window.TerraProject.path.datasets;
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
            )
        });

        ProjectNew.find(".actions-form > .cancel > button").bind("click", (event) => {
            event.preventDefault();
            ProjectNew.close();
        });

        ProjectSave.find("form").bind("submit", (event) => {
            event.preventDefault();
            let data = $(event.currentTarget).serializeObject();
            data.overwrite = data.overwrite !== undefined;
            window.ExchangeRequest(
                "project_save",
                (success, data) => {
                    if (success) {
                        ProjectSave.close();
                        window.TerraProject.name = data.data.name;
                        window.StatusBar.message(window.Messages.get("PROJECT_SAVED"), true);
                    } else {
                        window.StatusBar.message(data.error, false);
                    }
                },
                data
            );
        });

    });


})(jQuery);
