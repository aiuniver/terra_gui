"use strict";


(($) => {


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
        let _name = options.name || DEFAULT_NAME;
        let _hardware = options.hardware || DEFAULT_HARDWARE;
        let _datasets = options.datasets || {};
        let _tags = options.tags || {};
        let _dataset = options.dataset || "";
        let _task = options.task || "";
        let _model_name = options.model_name || "";
        let _layers = options.layers || {};
        let _layers_types = options.layers_types || [];
        let _optimizers = options.optimizers || [];
        let _callbacks = options.callbacks || {};

        Object.defineProperty(this, "error", {
            set: (value) => {
                _error = value;
            },
            get: () => {
                return _error;
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
            set: (value) => {
                _datasets = value;
            },
            get: () => {
                return _datasets;
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

        Object.defineProperty(this, "task", {
            set: (value) => {
                _task = value;
            },
            get: () => {
                return _task;
            }
        });

        Object.defineProperty(this, "task_name", {
            get: () => {
                if (_task) {
                    return _tags[_task];
                } else {
                    return undefined;
                }
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

        Object.defineProperty(this, "layers", {
            set: (value) => {
                _layers = value;
            },
            get: () => {
                return _layers;
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

    }


    $(() => {

        window.TerraProject = new TerraProject(window._terra_project);

        if (window.TerraProject.error) {
            window.StatusBar.message(window.TerraProject.error, false);
        }

        $("header > .user > .item > .menu > .group > .title").bind("click", (event) => {
            $(event.currentTarget).parent().toggleClass("hidden");
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
            item.toggleClass("active");
        });
        $(document).bind("click", (event) => {
            let item = $(event.target);
            if (!item.closest(".user > .item").length) {
                $("header > .user > .item").removeClass("active");
            }
        });

    });


})(jQuery);
