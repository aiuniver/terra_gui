"use strict";


(($) => {


    $.fn.extend({


        StatusBar: function(){

            if (!this.length) return this;

            this.message = (text, status) => {
                if (status === true) status = "success";
                else if (status === false) status = "error";
                else status = "processing";
                this.find(".message > .wrapper").html(`<span class="${status}">${text}</span>`);
            }

            this.message_clear = () => {
                this.find(".message > .wrapper").html("");
            }

            this.progress = (percent, message) => {
                this.find(".progress > .wrapper > i").width(`${percent}%`);
                this.find(".progress > .wrapper span").text(message);
            }

            this.progress_clear = () => {
                this.find(".progress > .wrapper > i").width(0);
                this.find(".progress > .wrapper span").text("");
            }

            this.clear = () => {
                this.message_clear();
                this.progress_clear();
            }

            return this;

        }


    });


    $(() => {

        window.StatusBar = $("footer").StatusBar();

    })


})(jQuery);
