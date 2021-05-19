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

        }


    })


    $(() => {

        $(".custom-scrollbar-wrapper").CustomScrollbar();

        $("select.jquery-ui-menuselect").selectmenu();

        $(window).bind("resize", (event) => {
            $("select.jquery-ui-menuselect").selectmenu("close");
        });

    });


})(jQuery);
