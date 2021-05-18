"use strict";


(($) => {


    const MW_CLASSNAME = ".modal-window-container";


    let ModalWindow = function(mw) {

        let _size = [0, 0],
            _title = "",
            _noclose = false;

        this.open = (title, size, noclose) => {
            this.title = title;
            this.size = size;
            mw.addClass("opened");
            mw.find(".inner > .container").addClass("loading");
            this.noclose = noclose === true;
        }

        this.opened = () => {
            mw.find(".inner > .container").removeClass("loading");
        }

        this.close = () => {
            if (this.noclose) return null;
            mw.removeClass("opened");
            $(MW_CLASSNAME).removeClass("visible");
            window.StatusBar.message_clear();
        }

        Object.defineProperty(this, "noclose", {
            set: (value) => {
                _noclose = value;
                let close = mw.find(".inner > .header > .close");
                _noclose ? close.addClass("hidden") : close.removeClass("hidden");
            },
            get: () => {
                return _noclose;
            }
        });

        Object.defineProperty(this, "size", {
            set: (value) => {
                _size = value;
                mw.children(".inner").css({
                    width:`${value[0]}px`,
                    height:`${value[1]}px`,
                    marginLeft:`-${value[0]/2}px`,
                    marginTop:`-${value[1]/2}px`,
                });
            },
            get: () => {
                return _size;
            }
        });

        Object.defineProperty(this, "title", {
            set: (value) => {
                _title = value;
                mw.find(".inner > .header > .title").text(value);
            }
        });

        mw.children(".overlay").bind("click", this.close);
        mw.find(".inner > .header > .close").bind("click", this.close);

        $(window).bind("keydown", (event) => {
            if (this.noclose) return null;
            if (event.keyCode === 27) this.close();
        });

    }


    $.fn.extend({


        ModalWindow: function(options) {

            if (!this.length) return this;

            this.addClass(MW_CLASSNAME);

            this.open = (callback) => {
                if (typeof callback === "function") options.callback = callback;
                window.ModalWindow.open(options.title, [options.width, options.height], options.noclose);
                $(MW_CLASSNAME).removeClass("visible");
                if (options.request) {
                    window.ExchangeRequest(
                        options.request[0],
                        (success, data) => {
                            if (success) {
                                this.addClass("visible");
                                window.ModalWindow.opened();
                                if (typeof options.callback === "function") options.callback(this, data.data);
                            } else {
                                window.StatusBar.message(data.error, false);
                            }
                        },
                        options.request[1]
                    );
                } else {
                    this.addClass("visible");
                    window.ModalWindow.opened();
                    if (typeof options.callback === "function") options.callback(this);
                }
            }

            this.close = () => {
                window.ModalWindow.close();
            }

            return this;

        }

    })


    $(() => {

        window.ModalWindow = new ModalWindow($("#modal-window"));

    })


})(jQuery);
