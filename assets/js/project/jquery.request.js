"use strict";


(($) => {


    $.extend({


        RequestAPI: function(url, callback, send_data){

            let request;

            this.abort = () => {
                if (request) request.abort();
            }

            let execute = () => {
                request = $.ajax({
                    url:url,
                    type:"POST",
                    data:JSON.stringify(send_data),
                    contentType:"application/json; charset=UTF-8",
                    success:(data, status) => {
                        if (status !== "success") {
                            if (typeof callback === "function") {
                                callback(false, {"error":window.Messages.get("INTERNAL_SERVER_ERROR")});
                            }
                            return;
                        }
                        if (!data.success) {
                            if (typeof callback === "function") {
                                callback(false, data);
                            }
                            return;
                        }
                        if (typeof callback === "function") {
                            callback(true, data);
                        }
                        if (data.stop_flag === false) {
                            setTimeout(execute, 1000);
                        }
                    },
                    error:(xhr) => {
                        if (xhr.status === 502) window.location.reload();
                        if (typeof callback === "function") {
                            callback(false, {"error":window.Messages.get("INTERNAL_SERVER_ERROR")});
                        }
                    }
                })
            }

            execute();

            return this;

        }


    })


    window.ExchangeRequest = (action, callback, send_data) => {
        if (!action) return;
        return new $.RequestAPI(`/api/v1/exchange/${action}/`, callback, send_data);
    }


})(jQuery);
