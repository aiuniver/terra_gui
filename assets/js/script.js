"use strict";


(($) => {


    $.fn.extend({


        CustomScrollbar: function() {

            if (!this.length) return this;

            let _init = (item) => {
                item.mCustomScrollbar({
                    axis:`y${item.data("horizontal") ? "x" : ""}`,
                    scrollbarPosition:"outside",
                    scrollInertia:0,
                });
            }

            return this.each((index, item) => {
                _init($(item));
            });

        },

        CollapsableGroup: function() {

            if (!this.length) return this;

            let _init = (item) => {
                item.children(".params-title").bind("click", (event) => {
                    event.preventDefault();
                    $(event.currentTarget).parent().toggleClass("collapsed");
                });
            }

            return this.each((index, item) => {
                _init($(item));
            });

        },

        AutoCompleteWidget: function() {

            if (!this.length) return this;

            return this.each((index, item) => {
                $(item).bind("focus", (event) => {
                    let item = $(event.currentTarget);
                    event.currentTarget.oldValue = item.val();
                    item.trigger("input");
                    item.addClass("onfocus");
                    item.select();
                }).bind("blur", (event) => {
                    let item = $(event.currentTarget);
                    item.removeClass("onfocus");
                    item.val(event.currentTarget.oldValue);
                }).autocomplete({
                    appendTo:$(item).parent(),
                    minLength:0,
                    delay:0,
                    source:item.dataset.source,
                    select:(event, ui) => {
                        event.target.oldValue = ui.item.id;
                        $(event.target).blur();
                    }
                });
            });

        }


    })


    $(() => {

        $(".custom-scrollbar-wrapper").CustomScrollbar();

        $("select.jquery-ui-menuselect").selectmenu();

        $("input.jquery-ui-autocomplete").AutoCompleteWidget();

        $(window).bind("resize", (event) => {
            $("select.jquery-ui-menuselect").selectmenu("close");
        });

    });


})(jQuery);
