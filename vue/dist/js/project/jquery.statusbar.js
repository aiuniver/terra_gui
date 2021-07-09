"use strict";


(($) => {


    let StatusBarError = $("#modal-window-status-bar-error").ModalWindow({
        title:"Ошибка!",
        width:680,
        height:440,
        no_clear_status_bar:true,
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

        
        StatusBar: function(){

            if (!this.length) return this;

            this.message = (text, status) => {
                if (status === true) status = "success";
                else if (status === false) status = "error";
                else status = "processing";
                this.find(".message > .wrapper").html(`<span class="${status}">${text}</span>`);
                if (status === "error") {
                    this.find(".message > .wrapper > .error").bind("click", (event) => {
                        StatusBarError.open((ui) => {
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
                    });
                }
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

        StatusBarError.find(".action > .clipboard").bind("click", (event) => {
            let result = StatusBarError.find(".action > .result"),
                pre = StatusBarError.find("pre");
            result.text("");
            if (fallbackCopyTextToClipboard(pre.text())) {
                result.text("Код скопирован в буфер обмена");
                clip(pre[0]);
            }
        });

    })


})(jQuery);
